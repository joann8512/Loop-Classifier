# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:24:34 2020

@author: Joann
"""
#%%
import torch
import torch.nn as nn
from torch.autograd import Variable
from modules import HarmonicSTFT

#%%

class Model(nn.Module):
    def __init__(self,
                conv_channels=128,
                sample_rate=16000,
                n_fft=513,
                n_harmonic=6,
                semitone_scale=2,
                learn_bw=None):  # bw = bandwidth
        super(Model, self).__init__()
        self.hstft = HarmonicSTFT(sample_rate=sample_rate,
                                    n_fft=n_fft,
                                    n_harmonic=n_harmonic,
                                    semitone_scale=semitone_scale,
                                    learn_bw=learn_bw)
        self.hstft_bn = nn.BatchNorm2d(n_harmonic)

        # 2D CNN
        from modules import ResNet_mtat as ResNet
        
        self.conv_2d = ResNet(input_channels=n_harmonic, conv_channels=conv_channels)

    def forward(self, x):
        # harmonic stft
        hstft = self.hstft(x)
        x = self.hstft_bn(hstft)

        # 2D CNN
        logits = self.conv_2d(x)

        return logits
