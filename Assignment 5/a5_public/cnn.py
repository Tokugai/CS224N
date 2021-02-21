#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn

class CNN(nn.Module):
    '''CNN
    - 1-dimensional convolution
    - Max Pooling
    '''
    def __init__(self, char_embed_size, word_embed_size, k=5):
        '''Init CNN module

        @param char_embed_size (int): size of character embedding
        @param word_embed_size (int): size of word embedding
        @param k (int): filter size
        '''
        super(CNN, self).__init__()
        self.char_embed_size = char_embed_size
        self.word_embed_size = word_embed_size
        self.k = k

        self.conv1D = nn.Conv1d(in_channels=char_embed_size, out_channels=word_embed_size,kernel_size=k, bias=True)
        self.relu = nn.ReLU()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Take the input character embedding, run through a convolutional layer
            and max pooling, and output to be fed into then highway network
        @param x (Tensor): input tensor of size (batch_size, word_embed_size, char_embed_size)
        
        @return x_conv_output (Tensor): model embedding output of size (batch_size, word_embed_size)
        '''
        x_conv = self.conv1D(x)
        x_conv_output, _ = torch.max(self.relu(x_conv),dim=2)

        return x_conv_output

### END YOUR CODE

