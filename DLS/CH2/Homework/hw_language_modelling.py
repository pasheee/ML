
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from datasets import load_dataset
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
import nltk

from collections import Counter
from typing import List

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.cuda.amp import GradScaler, autocast
from dataset import WordDataset

import seaborn
seaborn.set(palette='summer')
mp.set_start_method('spawn', force=True)
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"


# In[5]:


nltk.download('punkt')
nltk.download('punkt_tab')



device = torch.device("cuda")



dataset = load_dataset('imdb')




sentences = []
word_threshold = 32
train_dataset = dataset['train']
test_dataset = dataset['test']

for sample in train_dataset['text']:
    sentences.extend([sentence for sentence in sent_tokenize(sample) if len(sentence.split()) < word_threshold])


print("Всего предложений:", len(sentences))



words = Counter()

for sentence in tqdm(sentences):
    for word in word_tokenize(sentence):
        words[word] += 1




len(words.keys())



vocab = set()
vocab_size = 40000
vocab.update(['<unk>', '<bos>', '<eos>', '<pad>'])
count = 0
for word in words.keys():
    vocab.add(word)
    count+=1
    if count == 40000:
        break




assert '<unk>' in vocab
assert '<bos>' in vocab
assert '<eos>' in vocab
assert '<pad>' in vocab
assert len(vocab) == vocab_size + 4




print("Всего слов в словаре:", len(vocab))



word2ind = {char: i for i, char in enumerate(vocab)}
ind2word = {i: char for char, i in word2ind.items()}





def collate_fn_with_padding(
    input_batch: List[List[int]], pad_id=word2ind['<pad>']) -> torch.Tensor:
    seq_lens = [len(x) for x in input_batch]
    max_seq_len = max(seq_lens)

    new_batch = []
    for sequence in input_batch:
        for _ in range(max_seq_len - len(sequence)):
            sequence.append(pad_id)
        new_batch.append(sequence)

    sequences = torch.LongTensor(new_batch).to(device)

    new_batch = {
        'input_ids': sequences[:,:-1],
        'target_ids': sequences[:,1:]
    }

    return new_batch



if __name__ == "__main__":
    
    train_sentences, eval_sentences = train_test_split(sentences, test_size=0.2)
    eval_sentences, test_sentences = train_test_split(sentences, test_size=0.5)
    
    train_dataset = WordDataset(train_sentences, word2ind)
    eval_dataset = WordDataset(eval_sentences, word2ind)
    test_dataset = WordDataset(test_sentences, word2ind)
    
    batch_size = 8
    
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=collate_fn_with_padding,
        batch_size=batch_size,
        num_workers=4  
    )
    
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=collate_fn_with_padding, batch_size=batch_size)
    
    test_dataloader = DataLoader(
        test_dataset, collate_fn=collate_fn_with_padding, batch_size=batch_size)
    
    
    
    
    def evaluate(model, criterion, dataloader) -> float:
        model.eval()
        perplexity = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                logits = model(batch['input_ids'].to(device)).flatten(start_dim=0, end_dim=1)
                loss = criterion(logits, batch['target_ids'].to(device).flatten())
                perplexity.append(torch.exp(loss).item())
    
        perplexity = sum(perplexity) / len(perplexity)
    
        return perplexity
    
    
    
    from torch.amp import GradScaler, autocast
    
    
    def train_model(model, criterion, train_dataloader, eval_dataloader, optimizer, num_epoch):
        model.train()
        losses = []
        perplexities = []
        scaler = GradScaler('cuda') 
        for epoch in range(num_epoch):
            epoch_losses = []
            model.train()
            for batch in tqdm(train_dataloader, desc=f'Training epoch {epoch}:'):
                optimizer.zero_grad()
                with autocast('cuda'):
                    logits = model(batch['input_ids'].to(device)).flatten(start_dim=0, end_dim=1)
                    loss = criterion(logits, batch['target_ids'].to(device).flatten())
                scaler.scale(loss).backward()  
                scaler.step(optimizer)         
                scaler.update()   
                epoch_losses.append(loss.item())
    
            losses.append(sum(epoch_losses) / len(epoch_losses))
            perplexities.append(evaluate(model, criterion, eval_dataloader))
    
        return losses, perplexities
    
    
    
    
    class LanguageModel(nn.Module):
        def __init__(self, hidden_dim: int, vocab_size: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_dim)
            self.rnn = nn.RNN(hidden_dim, hidden_dim, batch_first=True)
            self.linear = nn.Linear(hidden_dim, hidden_dim)
            self.projection = nn.Linear(hidden_dim, vocab_size)
    
            self.non_lin = nn.Tanh()
            self.dropout = nn.Dropout(p=0.1)
    
        def forward(self, input_batch) -> torch.Tensor:
            embeddings = self.embedding(input_batch)  # [batch_size, seq_len, hidden_dim]\
            output, _ = self.rnn(embeddings)  # [batch_size, seq_len, hidden_dim]
            output = self.dropout(self.linear(self.non_lin(output)))  # [batch_size, seq_len, hidden_dim]
            projection = self.projection(self.non_lin(output))  # [batch_size, seq_len, vocab_size]
    
            return projection
    
    
    
    
    
    model = LanguageModel(hidden_dim=256, vocab_size=len(vocab))
    model = model.to(device)  
    criterion = nn.CrossEntropyLoss(ignore_index=word2ind['<pad>'])
    optimizer = torch.optim.Adam(model.parameters())
    
    
    losses, perplexities = train_model(model, criterion, train_dataloader, eval_dataloader, optimizer, 5)




    plt.plot(losses)
    plt.show()
    
    
    
    
    plt.plot(perplexities)
    plt.show()
    
    
    
    
    class LanguageModel(nn.Module):
        def __init__(self, hidden_dim: int, vocab_size: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_dim)
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, dropout = 0.1)
            self.linear = nn.Linear(hidden_dim, hidden_dim)
            self.projection = nn.Linear(hidden_dim, vocab_size)
    
            self.non_lin = nn.Tanh()
            self.dropout = nn.Dropout(p=0.1)
    
        def forward(self, input_batch) -> torch.Tensor:
            embeddings = self.embedding(input_batch)  # [batch_size, seq_len, hidden_dim]\
            output, _ = self.rnn(embeddings)  # [batch_size, seq_len, hidden_dim]
            output = self.dropout(self.linear(self.non_lin(output)))  # [batch_size, seq_len, hidden_dim]
            projection = self.projection(self.non_lin(output))  # [batch_size, seq_len, vocab_size]
    
            return projection
    
    model = LanguageModel(hidden_dim=256, vocab_size=len(vocab)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=word2ind['<pad>'])
    optimizer = torch.optim.Adam(model.parameters())
    
    losses, perplexities = train_model(model, criterion, train_dataloader, optimizer, 5)
    
    
    
    
    plt.plot(losses)
    plt.show()
    
    
    
    plt.plot(perplexities)
    plt.show()






