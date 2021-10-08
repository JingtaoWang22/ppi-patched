#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 23:33:28 2021

@author: jingtao
"""
from torchvision.transforms import ToTensor, Lambda
from utils import data_loader
from preprocessor import ppi_preprocessor
import numpy as np
from models.convmixer import *
import torch
import time
import math
from torch.utils.data import DataLoader




''' training hyperparameters'''
batch_size = 16
warmup_epochs = 50
train_epochs = 100
learning_rate = 1e-4
decay_rate = 0.5
decay_interval = 10
epochs = 10


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = torch.tensor(X)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")




if __name__ == "__main__":    

    '''device'''
    if torch.cuda.is_available:
        device = 'cuda'
        print('using gpu')
    else:
        device = 'cpu'
        print('using cpu')
    
    '''preprocess'''
    #p=ppi_preprocessor()
    #p.ngram_dic_dataset()

    
    ''' data '''
    loader=data_loader()
    training_data,test_data, word_dict=loader.load()
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    ''' choose a model. '''
    #model = convmixer(len(word_dict),dim=10 )
    model = ConvMixer(dim=10, depth=6, kernel_size=9, patch_size=7, n_classes=2)
    model = model.to(device)
    model2 = PPIConvMixer(dim=10, depth=6, kernel_size=9, embedding_dim=10, n_words=len(word_dict),\
                          patch_size=7, n_classes=2)
    model2 = model2.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    
    #epochs = warmup_epochs+train_epochs  
    #epoch_lr = 0
    
    
    
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")
    
    
    
    
    
    
    
    
    

'''
    for i in range(n_updates):  
        (x) = next(iter(dataloader))
        x = x.to(device)
        #x=x.float()
    
        #x = x.reshape([batch_size,1,28,28])
        #x = x.reshape((batch_size,1,120,200))
        optimizer.zero_grad()

        embedding_loss, x_hat, perplexity, quantum = model(x)
        recon_loss = torch.mean((x_hat - x)**2) / x_train_var
        loss = recon_loss + embedding_loss

        loss.backward()
        optimizer.step()

        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["n_updates"] = i

        if i % log_interval == 0:
            """
            save model and print values
            """
            if save:
                hyperparameters = 'hyp'
                utils.save_model_and_results(
                        model, results, hyperparameters, filename)

            print('Update #', i, 'Recon Error:',
                  np.mean(results["recon_errors"][-log_interval:]),
                  'Loss', np.mean(results["loss_vals"][-log_interval:]),
                  'Perplexity:', np.mean(results["perplexities"][-log_interval:]))

'''

