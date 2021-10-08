#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 23:34:25 2021

@author: jingtao
"""


import numpy as np
import pickle
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

class ppidataloader(Dataset):
    def __init__(self, ppix, ppiy, transform=None, target_transform=None):
        self.xs = ppix
        self.ys = ppiy
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        #if self.transform:
        #    image = self.transform(image)
        #if self.target_transform:
        #    label = self.target_transform(label)
        return x, y

class data_loader:
    
    def __init__(self):
        return
    
    def split_dataset(self, dataset, ratio):
        n = int(ratio * len(dataset))
        dataset_1, dataset_2 = dataset[:n], dataset[n:]
        return dataset_1, dataset_2

    def load_pickle(self, file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    
    def load(self, path='./data/preprocessed/',data='yeast.npy', dic='yeast_dic.pickle', augmentation=False):
        
        
        dataset = np.load(path+data,allow_pickle=True)
        max_len=int(dataset[-1])
        dataset=dataset[:-1]

        train, test = self.split_dataset(dataset, 0.8)
        
        x_train=[]
        y_train=[]
        x_test=[]
        y_test=[]
        
        for sample in train:

            p1 = np.zeros(max_len)
            p1[0:sample[0].shape[0]]+=sample[0]
            p2 = np.zeros(max_len)
            p2[0:sample[1].shape[0]]+=sample[1]
            pp=np.array((p1,p2))
            x_train.append(pp)
            
            #y = np.zeros((2,))
            #y[np.array(sample[2])]=1
            ##y_train.append(y.astype('int'))
            
            y_train.append(sample[2])
            
            #if (augmentation ==True):
            #    x_train.append((sample[1].astype('float'),sample[0].astype('float')))
            #    y_train.append(float(sample[2][0]))
                
        for sample in test:

            p1 = np.zeros(max_len)
            p1[0:sample[0].shape[0]]+=sample[0]
            p2 = np.zeros(max_len)
            p2[0:sample[1].shape[0]]+=sample[1]
            pp=np.array((p1,p2))
            
            x_test.append(pp)
            
            #y = np.zeros((2,))
            #y[np.array(sample[2])]=1
            #y_test.append(y.astype('int'))
            y_test.append(sample[2])
        
        
        word_dict = self.load_pickle(path+dic)

        
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        
        train_loader = ppidataloader(x_train,y_train)
        test_loader = ppidataloader(x_test,y_test)
        return train_loader, test_loader, word_dict


'''
loader=data_loader()
x_train,y_train,x_test,y_test = loader.load()
'''