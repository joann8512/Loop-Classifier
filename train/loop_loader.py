# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:53:48 2020

@author: Joann
"""
#%%
import os
import numpy as np
from torch.utils import data
#%%

class AudioFolder(data.Dataset):
    def __init__(self, root, input_length=None):
        self.root = root  # data_path
        self.input_length = input_length  # 80000
        self.get_songlist()  # training data npy
        self.binary = np.load(os.path.join(self.root, 'binary_full_1sec.npy'))

    def __getitem__(self, index):
        #print("INDEX:", index)
        npy, tag_binary = self.get_npy(index)
        return npy.astype('float32'), tag_binary.astype('float32')

    def get_songlist(self):
        self.fl = np.load(os.path.join(self.root, 'train_full_1sec.npy'))  # npy with all training file names

    def get_npy(self, index):
        ix, fn = self.fl[index].split('\t')  # fn = wav filename
        npy_path = os.path.join(self.root, 'npy_full_1sec', fn.split('.')[0]+'.npy')  #.split('/')[1][:-3])  # only get training files from npy
        npy = np.load(npy_path)
        random_idx = int(np.floor(np.random.random(1) * (len(npy)-self.input_length)))  # input_length = how long to go in model
        npy = np.array(npy[random_idx:random_idx+self.input_length])
        tag_binary = self.binary[int(ix)]
        return npy, tag_binary

    def __len__(self):
        return len(self.fl)

#%%

def get_audio_loader(root, batch_size, num_workers=0, input_length=None):
    data_loader = data.DataLoader(dataset=AudioFolder(root, input_length=input_length),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=num_workers)
    return data_loader

