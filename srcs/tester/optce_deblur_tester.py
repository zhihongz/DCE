import logging
import os
import cv2
import torch
import time
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm
from srcs.utils.util import instantiate
from srcs.utils.utils_image_kair import tensor2uint, imsave
from srcs.utils.utils_patch_proc import window_partitionx, window_reversex
from srcs.model.deeprft_utils import load_checkpoint_compress_doconv, load_checkpoint
import torch.nn.functional as F
from srcs.utils.utils_eval_zzh import gpu_inference_time_est


def testing(gpus, config):
    test_worker(gpus, config)


def test_worker(gpus, config):
    # prevent access to non-existing keys
    OmegaConf.set_struct(config, True)

    # logger & dir setting
    logger = logging.getLogger('test')
    if not os.path.exists(config.outputs_dir):
        os.makedirs(config.outputs_dir)

    # prepare model & checkpoint for testing
    # load checkpoint
    logger.info('Loading checkpoint: {} ...'.format(config.checkpoint))
    checkpoint = torch.load(config.checkpoint)
    if 'config' in checkpoint:
        loaded_config = OmegaConf.create(checkpoint['config'])
    else:
        loaded_config = config

    # inference & test_sigma_setting
    loaded_config.inference = True

    # instantiate model
    model = instantiate(loaded_config.arch)
    logger.info(model)
    if len(gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpus)
    
    # state_dict = checkpoint['state_dict']
    # model.load_state_dict(state_dict)
    # load_checkpoint(model, config.checkpoint) # for deeprft
    load_checkpoint_compress_doconv(model, config.checkpoint)  # for deeprft



    # reset param
    model.BlurNet.test_sigma_range = config.test_sigma_range
    
    # instantiate loss and metrics
    criterion = instantiate(loaded_config.loss, is_func=False)
    metrics = [instantiate(met, is_func=True) for met in loaded_config.metrics]

    # setup data_loader instances
    data_loader = instantiate(config.test_data_loader)

    # test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log = test(data_loader, model,
               device, criterion, metrics, config)
    logger.info(log)


def test(data_loader, model,  device, criterion, metrics, config):
    '''
    test step
    '''

    # init
    model = model.to(device)

    # inference time test
    # input_shape = (1, 32, 3, 256, 256)  # test image size
    # gpu_inference_time_est(model, input_shape)

    ce_weight = model.BlurNet.ce_weight.detach().squeeze()
    ce_code = ((torch.sign(ce_weight)+1)/2).int()

    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metrics))
    time_start = time.time()
    with torch.no_grad():
        for i, vid in enumerate(tqdm(data_loader, desc='Testing')):
            vid = vid.to(device)
            target = vid[:, vid.shape[1]//2, ...]

            N, F, C, Hx, Wx = vid.shape

            # direct
            # data, output = model(vid)

            # sliding window - patch processing
            vid = vid.permute(1, 0, 2, 3, 4)
            vid_ = []
            for k in range(F):
                tmp, batch_list = window_partitionx(vid[k], config.win_size)
                vid_.append(tmp.unsqueeze(0))
            vid = torch.cat(vid_, dim=0)
            vid = vid.permute(1, 0, 2, 3, 4)
            data_, output_ = model(vid)
            data = window_reversex(
                data_, config.win_size, Hx, Wx, batch_list)
            output = window_reversex(
                output_, config.win_size, Hx, Wx, batch_list)

            # clamp to 0-1
            output = torch.clamp(output, 0, 1)

            # save some sample images
            scale_fc = len(ce_code)/sum(ce_code)
            for k, (in_img, out_img, gt_img) in enumerate(zip(data, output, target)):
                in_img = tensor2uint(in_img*scale_fc)
                out_img = tensor2uint(out_img)
                gt_img = tensor2uint(gt_img)
                imsave(
                    in_img, f'{config.outputs_dir}test{i+1:02d}_{k+1:04d}_in_img.jpg')
                imsave(
                    out_img, f'{config.outputs_dir}test{i+1:02d}_{k+1:04d}_out_img.jpg')
                imsave(
                    gt_img, f'{config.outputs_dir}test{i+1:02d}_{k+1:04d}_gt_img.jpg')
                # break  # save one image per batch

            # computing loss, metrics on test set
            loss = criterion(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metrics):
                total_metrics[i] += metric(output, target) * batch_size
    time_end = time.time()
    time_cost = time_end-time_start
    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples,
           'time/sample': time_cost/n_samples,
           'ce_code': ce_code}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metrics)
    })
    return log
