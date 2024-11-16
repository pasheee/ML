#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch.nn as nn
import torch

class Translator(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, hidden_dim, dropout_prob = 0):
        super().__init__()
        self.source_embeddings = nn.Embedding(source_vocab_size, hidden_dim)
        self.target_embeddings = nn.Embedding(target_vocab_size, hidden_dim)

        
        nn.init.xavier_uniform_(self.source_embeddings.weight, nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.target_embeddings.weight, nn.init.calculate_gain('tanh'))

        
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first = True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first = True)

        
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.projection = nn.Linear(hidden_dim, target_vocab_size)

        self.non_lin = nn.Tanh()
        self.dropout = nn.Dropout(p = dropout_prob)

    def forward(self, source, target):
        source_embeddings = self.source_embeddings(source)
        target_embeddings = self.target_embeddings(target)

        _, encoded_hidden_vector = self.encoder(source_embeddings)

        output, _ = self.decoder(target_embeddings, encoded_hidden_vector) 
        linear_output = self.dropout(self.non_lin(self.linear(output)))
        
        projection = self.projection(linear_output) 

        return projection

class Translatorv2(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, hidden_dim, dropout_prob=0, nhead=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Проверяем, что hidden_dim делится на количество голов
        assert hidden_dim % nhead == 0, "hidden_dim должен быть кратен количеству голов attention"
        
        self.source_embeddings = nn.Embedding(source_vocab_size, hidden_dim)
        self.target_embeddings = nn.Embedding(target_vocab_size, hidden_dim)
        
        nn.init.xavier_uniform_(self.source_embeddings.weight, nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.target_embeddings.weight, nn.init.calculate_gain('tanh'))
        
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Добавляем MultiheadAttention
        self.attention = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout_prob, batch_first=True)
        
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.projection = nn.Linear(hidden_dim, target_vocab_size)
        
        self.non_lin = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, source, target):
        # Получаем эмбеддинги
        source_embeddings = self.source_embeddings(source)
        target_embeddings = self.target_embeddings(target)
        
        # Энкодим source последовательность
        encoder_outputs, (hidden, cell) = self.encoder(source_embeddings)
        
        # Декодим target последовательность
        decoder_outputs, _ = self.decoder(target_embeddings, (hidden, cell))
        
        # Применяем механизм внимания
        # query - это выход декодера
        # key и value - это выход энкодера
        attn_output, _ = self.attention(
            decoder_outputs,  # query
            encoder_outputs,  # key
            encoder_outputs   # value
        )
        
        # Применяем dropout и нелинейность
        linear_output = self.dropout(self.non_lin(self.linear(attn_output)))
        
        # Проекция на размер целевого словаря
        projection = self.projection(linear_output)
        
        return projection
