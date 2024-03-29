# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:23:01 2020

@author: Joann
"""

import os
import argparse
from solver import Solver

#%%

def main(config):
    # path for models
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)

    # import data loader
    from loader import get_audio_loader

    # get data loader
    data_loader = get_audio_loader(config.data_path,
                                   config.batch_size,  # 16
                                   input_length=config.input_length,  # 80000
                                   num_workers=config.num_workers)  # 4

    solver = Solver(data_loader, config)

    print('Solver.train')
    solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # model hyper-parameters
    parser.add_argument('--conv_channels', type=int, default=128)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--n_fft', type=int, default=513)
    parser.add_argument('--n_harmonic', type=int, default=6)
    parser.add_argument('--semitone_scale', type=int, default=2)
    parser.add_argument('--learn_bw', type=str, default='only_Q', choices=['only_Q', 'fix'])

    # dataset
    parser.add_argument('--input_length', type=int, default=48000)
    parser.add_argument('--num_workers', type=int, default=4)

    # training settings
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_tensorboard', type=int, default=1)
    parser.add_argument('--model_save_path', type=str, default='./models_full')
    parser.add_argument('--model_load_path', type=str, default='.')
    parser.add_argument('--data_path', type=str, default='./Loop-Classifier/')
    parser.add_argument('--log_step', type=int, default=20)

    config = parser.parse_args()

    print(config)
    main(config)

