import numpy as np
import os
import hydra
import torch
import warnings
from omegaconf import OmegaConf
from importlib import import_module
# torch.autograd.set_detect_anomaly(True) # for debug

# fix random seeds for reproducibility
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ignore warning
warnings.filterwarnings('ignore')


@hydra.main(config_path='conf/', config_name='optce_deblur_train')
def main(config):
    # GPU setting
    if not config.gpus or config.gpus == -1:
        gpus = list(range(torch.cuda.device_count()))
    else:
        gpus = config.gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
    n_gpu = len(gpus)
    assert n_gpu <= torch.cuda.device_count(
    ), 'Can\'t find %d GPU device on this machine.' % (n_gpu)

    # resume
    config_v = OmegaConf.to_yaml(config, resolve=True)

    # show config
    print('='*40+'\n', config_v, '\n'+'='*40+'\n')

    # training
    trainer_name = 'srcs.trainer.%s' % config.trainer_name
    training_module = import_module(trainer_name)
    training_module.trainning(gpus, config)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
