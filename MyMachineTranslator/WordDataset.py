#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
from torch.utils.data import Dataset




class WordDataset(Dataset):
    def __init__(self, source, target, source_word2ind, target_word2ind, source_max_len, target_max_len):
        self.source_samples = source
        self.target_samples = target

        self.source_word2ind = source_word2ind
        self.target_word2ind = target_word2ind
        
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        
    def __len__(self):
        return len(self.source_samples)


    def __getitem__(self, idx):
        source_sentence = self.source_samples[idx][:self.source_max_len]
        target_sentence = self.target_samples[idx][:self.target_max_len]
        
        # Кодирование предложений
        source_indices = [self.source_word2ind['<SOS>']] + \
                         [self.source_word2ind.get(word, self.source_word2ind['<UNK>']) for word in source_sentence] + \
                         [self.source_word2ind['<EOS>']]
        
        target_indices = [self.target_word2ind['<SOS>']] + \
                         [self.target_word2ind.get(word, self.target_word2ind['<UNK>']) for word in target_sentence] + \
                         [self.target_word2ind['<EOS>']]
        
        # Добавляем паддинг
        source_indices += [self.source_word2ind['<PAD>']] * (self.source_max_len + 2 - len(source_indices))
        target_indices += [self.target_word2ind['<PAD>']] * (self.target_max_len + 2 - len(target_indices))
        
        
        # Преобразование в тензоры
        source_tensor = torch.tensor(source_indices, dtype=torch.long)
        target_tensor = torch.tensor(target_indices, dtype=torch.long)
    
        return source_tensor, target_tensor



