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

    # load weight
    # state_dict = checkpoint['state_dict']
    # model.load_state_dict(state_dict)
    # load_checkpoint(model, config.checkpoint) # for deeprft
    load_checkpoint_compress_doconv(model, config.checkpoint)  # for deeprft

    # instantiate loss and metrics

    metrics = [instantiate(met, is_func=True) for met in loaded_config.metrics]

    # setup data_loader instances
    data_loader = instantiate(config.test_data_loader)

    # test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log = test(data_loader, model,
               device, metrics, config)
    logger.info(log)


def test(data_loader, model,  device, metrics, config):
    '''
    test step
    '''

    # init
    model = model.to(device)
    ce_weight = model.BlurNet.ce_weight.detach().squeeze()
    ce_code = ((torch.sign(ce_weight)+1)/2).int()
    scale_fc = len(ce_code)/sum(ce_code)

    # extract deblur model
    model_deblur = model.DeBlurNet  # deblur model

    # inference time test
    # input_shape = (1, 32, 3, 256, 256)  # test image size
    # gpu_inference_time_est(model, input_shape)
    # starter, ender = torch.cuda.Event(
    #         enable_timing=True), torch.cuda.Event(enable_timing=True)

    # ce_weight = model.BlurNet.ce_weight.detach().squeeze()
    # ce_code = ((torch.sign(ce_weight)+1)/2).int()

    model_deblur.eval()
    total_loss = 0.0
    total_metrics = torch.zeros(len(metrics))
    time_start = time.time()
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader, desc='Testing')):
            data = torch.flip(data.to(device), [2, 3])
            _data = data/scale_fc

            _, _, Hx, Wx = data.shape
            # sliding window - patch processing
            # starter.record()
            data_re, batch_list = window_partitionx(_data, config.win_size)
            output = model_deblur(data_re)
            output = window_reversex(
                output, config.win_size, Hx, Wx, batch_list)
            # ender.record()
            torch.cuda.synchronize()
            # time_cost = starter.elapsed_time(ender)
            # print(f"GPU inference time: {time_cost} ms")

            # pad & crop
            # sf = 4
            # HX, WX = int((Hx+sf-1)/sf)*sf, int((Wx+sf-1)/sf) * \
            #     sf  # pad to a multiple of scale_factor (sf)
            # pad_h, pad_w = HX-Hx, WX-Wx
            # data_pad = F.pad(_data/scale_fc, [0, pad_w, 0, pad_h])
            # output = model_deblur(data_pad)
            # output = output[:, :, :Hx, :Wx]

            # clamp to 0-1
            output = torch.clamp(output, 0, 1)

            # save some sample images
            for k, (in_img, out_img) in enumerate(zip(data, output)):
                in_img = tensor2uint(in_img)
                out_img = tensor2uint(out_img)
                imsave(
                    in_img, f'{config.outputs_dir}test{i+1:02d}_{k+1:04d}_in_img.jpg')
                imsave(
                    out_img, f'{config.outputs_dir}test{i+1:02d}_{k+1:04d}_out_img.jpg')

    time_end = time.time()
    time_cost = time_end-time_start
    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples,
           'time/sample': time_cost/n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metrics)
    })
    return log
