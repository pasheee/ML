#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch.nn as nn

class Translator(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, hidden_dim, dropout_prob = 0):
        super().__init__()
        self.source_embeddings = nn.Embedding(source_vocab_size, hidden_dim)
        self.target_embeddings = nn.Embedding(target_vocab_size, hidden_dim)

        nn.init.xavier_uniform_(self.source_embeddings.weight, nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.target_embeddings.weight, nn.init.calculate_gain('tanh'))
        
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first = True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first = True)
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.projection = nn.Linear(2*hidden_dim, target_vocab_size)

        self.non_lin = nn.Tanh()
        self.dropout = nn.Dropout(p = dropout_prob)

    def forward(self, source, target):
        source_embeddings = self.source_embeddings(source)
        target_embeddings = self.target_embeddings(target)

        _, encoded_hidden_vector = self.encoder(source_embeddings)
        output, _ = self.decoder(target_embeddings, encoded_hidden_vector) 
        output = self.non_lin(self.linear(self.non_lin(output))) 

        projection = self.projection(output) 

        return projection
        

