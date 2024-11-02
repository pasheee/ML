#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
from torch.utils.data import Dataset


class WordDataset(Dataset):
    def __init__(self, source, target, source_word2ind, target_word2ind, max_len = 50):
        
        self.source_samples = source
        self.target_samples = target

        self.source_word2ind = source_word2ind
        self.target_word2ind = target_word2ind
        self.max_len = max_len + 1

    def __len__(self):
        return len(self.source_samples)

    def __getitem__(self, idx):
        source_sentence = self.source_samples[idx]
        target_sentence = self.target_samples[idx]

        source_indices = [self.source_word2ind['<SOS>']] + [self.source_word2ind.get(word, self.source_word2ind['<UNK>']) for word in source_sentence]
        target_indices = [self.target_word2ind['<SOS>']] + [self.target_word2ind.get(word, self.target_word2ind['<UNK>']) for word in target_sentence]

        if len(source_indices) < self.max_len:
            source_indices.extend([self.source_word2ind['<PAD>'] for _ in range(self.max_len - len(source_indices))])
        else:
            source_indices = source_indices[:self.max_len]
        if len(target_indices) < self.max_len:
            target_indices.extend([self.target_word2ind['<PAD>'] for _ in range(self.max_len - len(target_indices))])
        else:
            target_indices = target_indices[:self.max_len]

        source_indices += [self.source_word2ind['<EOS>']]
        target_indices += [self.target_word2ind['<EOS>']]
        
        source_tensor = torch.tensor(source_indices, dtype = torch.long)
        target_tensor = torch.tensor(target_indices, dtype = torch.long)

        return source_tensor, target_tensor
        

