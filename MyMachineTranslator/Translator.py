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

        
        # nn.init.xavier_uniform_(self.source_embeddings.weight, nn.init.calculate_gain('tanh'))
        # nn.init.xavier_uniform_(self.target_embeddings.weight, nn.init.calculate_gain('tanh'))

        
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
    def __init__(self, source_vocab_size, target_vocab_size, hidden_dim, num_layers=1, n_heads=8, dropout_prob=0.3):
        super(Translatorv2, self).__init__()

        # Embeddings
        self.source_embeddings = nn.Embedding(source_vocab_size, hidden_dim)
        self.target_embeddings = nn.Embedding(target_vocab_size, hidden_dim)

        # nn.init.xavier_uniform_(self.source_embeddings.weight, nn.init.calculate_gain('tanh'))
        # nn.init.xavier_uniform_(self.target_embeddings.weight, nn.init.calculate_gain('tanh'))

        # Encoder: Bidirectional LSTM
        self.encoder = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )

        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout_prob, batch_first=True
        )

        # Decoder
        self.decoder = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )

        # Linear layers
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.projection = nn.Linear(hidden_dim, target_vocab_size)

        self.dropout = nn.Dropout(dropout_prob)
        self.non_lin = nn.Tanh()

    def forward(self, source, target, source_mask=None, target_mask=None):
        source_embeddings = self.source_embeddings(source)
        target_embeddings = self.target_embeddings(target)
    
        encoder_outputs, (hidden, cell) = self.encoder(source_embeddings)
    
        # Attention mechanism
        attn_output, _ = self.attention(
            query=target_embeddings,
            key=encoder_outputs,
            value=encoder_outputs,
            key_padding_mask=~source_mask  # Mask out PAD tokens
        )
    
        # Decoder
        decoder_outputs, _ = self.decoder(attn_output, (hidden, cell))
        
        projection = self.projection(decoder_outputs)
        
        # decoder_outputs = self.dropout(self.non_lin(self.linear(decoder_outputs)))
    
        # projection = self.projection(decoder_outputs)
        return projection





class Translatorv3(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, sp, tp, hidden_dim=512, n_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, dropout=0.1):
        super(Translatorv3, self).__init__()
        self.n_heads = n_heads
        self.target_embeddings = nn.Embedding(target_vocab_size, hidden_dim)
        self.source_embeddings = nn.Embedding(source_vocab_size, hidden_dim)
        
        nn.init.xavier_uniform_(self.source_embeddings.weight, nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.target_embeddings.weight, nn.init.calculate_gain('relu'))
        
        self.sp = sp
        self.tp = tp
        
        self.transformer = nn.Transformer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            dropout=dropout, 
            batch_first=True
        )
        
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.projection = nn.Linear(hidden_dim, target_vocab_size)
        
        self.non_lin = nn.ReLU()
        self.normalization = nn.LayerNorm(hidden_dim)

    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)
        return mask

    def create_padding_mask(self, seq, pad_idx):
        return (seq == pad_idx).to(dtype=torch.bool)

    
    def forward(self, source, target):
        # Transform inputs through embeddings
        target_embeddings = self.target_embeddings(target)
        source_embeddings = self.source_embeddings(source)
        
        tgt_seq_len = target.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(target.device)
        src_key_padding_mask = self.create_padding_mask(source, self.sp).to(source.device)
        tgt_key_padding_mask = self.create_padding_mask(target, self.tp).to(target.device)
        
        # Pass through transformer
        output = self.transformer(
            src=source_embeddings,
            tgt=target_embeddings,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        # Subsequent operations
        # output = self.non_lin(self.normalization(self.linear(output)))
        projection = self.projection(self.normalization(output))
        
        return projection


    
    






