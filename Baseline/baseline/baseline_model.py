import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from baseline_modules import HarmonicSTFT

#%%

class Model(nn.Module):
    def __init__(self,
                conv_channels=128,
                sample_rate=16000,
                n_fft=513,#513
                n_harmonic=1,
                semitone_scale=2,
                learn_bw=None):  # bw = bandwidth
        super(Model, self).__init__()
        self.stft = HarmonicSTFT(sample_rate=sample_rate,
                                    n_fft=n_fft,
                                    n_harmonic=n_harmonic,
                                    semitone_scale=semitone_scale,
                                    learn_bw=learn_bw)
        self.stft_bn = nn.BatchNorm2d(n_harmonic)

        # 2D CNN
        from baseline_modules import ResNet_mtat as ResNet
        
        self.conv_2d = ResNet(input_channels=n_harmonic, conv_channels=conv_channels)

    def forward(self, x):   
        stft = self.stft(x)
        x = self.stft_bn(stft)
        
        # 2D CNN
        logits = self.conv_2d(x)

        return logits
