#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 14:05:28 2021

@author: jingtao
"""

import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=2):
    return nn.Sequential(
        nn.Conv1d(1, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm1d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv1d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm1d(dim)
                )),
                nn.Conv1d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm1d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool1d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )
                
def PPIConvMixer(dim, depth, embedding_dim, n_words, kernel_size=9, patch_size=7, n_classes=2):
    return nn.Sequential(
        nn.Embedding(num_embeddings = n_words, embedding_dim = dim),
        nn.Conv1d(1, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm1d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv1d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm1d(dim)
                )),
                nn.Conv1d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm1d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool1d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )



