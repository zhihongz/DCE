import torch.nn as nn
import torch.nn.functional as F
from srcs.model.deeprft_model import DeepRFT
from srcs.model.optce_blur_model import CEBlurNet


class OptceDeblurNet(nn.Module):
    '''
    Fourier space wienner deblur network
    '''

    def __init__(self, sigma_range=0, test_sigma_range=None, ce_code_n=32, frame_n=64, ce_code_init=None, opt_cecode=False, blur_net=None, binary_fc=None, deblur_net=None, inference=False):
        super(OptceDeblurNet, self).__init__()
        # coded exposure blur net
        if blur_net == 'CEBlurNet':
            self.BlurNet = CEBlurNet(
                sigma_range=sigma_range, test_sigma_range=test_sigma_range, ce_code_n=ce_code_n, frame_n=frame_n, ce_code_init=ce_code_init, opt_cecode=opt_cecode, binary_fc=binary_fc)
        else:
            raise NotImplementedError

        # deep deblur net
        if deblur_net == 'DeepRFT':
            self.DeBlurNet = DeepRFT(inference=inference)
        else:
            raise NotImplementedError

    def forward(self, frames):
        ce_blur_img_noisy = self.BlurNet(frames)
        output = self.DeBlurNet(ce_blur_img_noisy)
        return ce_blur_img_noisy, output
