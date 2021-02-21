#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn

class Highway(nn.Module):
    '''
    Highway Layer

    '''
    def __init__(self, word_embed_size, dropout_rate=0.2):
        '''Init Highway module

        @param word_embed_size (int): Input/Output embedding size
        @ dropout_rate (float): Dropout probability used on final word embedding
        '''
        super(Highway, self).__init__()
        self.word_embed_size = word_embed_size
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(self.word_embed_size, self.word_embed_size)
        self.gate = nn.Linear(self.word_embed_size, self.word_embed_size)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x_conv: torch.Tensor) -> torch.Tensor:
        '''
        Take an input from a CNN, and run it through a Highway network to
        produce a word embedding

        @param x_conv (Tensor): tensor of dimension (batch_size, word_embedding_size)
            the output of the CNN

        @returns x_word_embed(Tensor): tensor of dimension (batch_size, word_embedding_size)
            the word embedding
        '''
        x_proj = self.relu(self.proj(x_conv))
        x_gate = self.sigmoid(self.gate(x_conv))
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv
        x_word_embed = self.dropout(x_highway)
        return x_word_embed

### END YOUR CODE 

