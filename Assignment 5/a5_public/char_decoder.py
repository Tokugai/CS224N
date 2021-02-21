#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()

        v_char = len(target_vocab.char2id)
        pad_token_idx = target_vocab.char2id['<pad>']

        self.charDecoder = nn.LSTM(input_size=char_embedding_size, hidden_size=hidden_size)
        self.char_output_projection = nn.Linear(in_features=hidden_size,out_features=v_char, bias=True)
        self.decoderCharEmb = nn.Embedding(num_embeddings=v_char,embedding_dim=char_embedding_size,padding_idx=pad_token_idx)
        self.target_vocab = target_vocab

        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        X = self.decoderCharEmb(input)
        output_x, dec_hidden = self.charDecoder(X, dec_hidden)
        scores = self.char_output_projection(output_x)
        
        return scores, dec_hidden
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        input = char_sequence[:-1,:]
        scores, dec_hidden = self.forward(input, dec_hidden)
        target = char_sequence[1:,:]

        [_, _, v_char] = list(scores.size())
        scores = scores.contiguous().view(-1, v_char)
        target = target.contiguous().view(-1)

        padding_idx = self.target_vocab.char2id['<pad>']
        loss_func = nn.CrossEntropyLoss(ignore_index=padding_idx, reduction='sum')
        loss = loss_func(scores, target)

        return loss
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        [_, batch, hidden_size] = list(initialStates[0].size())
        output = [""] * batch

        input_batch = torch.tensor([self.target_vocab.start_of_word] * batch, device=device)
        states = initialStates

        for i in range(max_length):
            input_batch = torch.unsqueeze(input_batch, 0)
            (scores, states) = self.forward(input_batch, states)
            scores = scores.squeeze(0)

            softmax_scores = nn.functional.softmax(scores, dim=1)
            input_batch = torch.max(softmax_scores, 1)[1]

            max_i = input_batch.data.tolist()

            max_char = [self.target_vocab.id2char[i] for i in max_i]

            output = [first + second for first, second in zip(output, max_char)]
        eow = self.target_vocab.id2char[self.target_vocab.end_of_word]
        for i, word in enumerate(output):
            last_i = word.find(eow)
            last_i = max_length if last_i == -1 else last_i
            output[i] = word[:last_i]

        return output

        ### END YOUR CODE

