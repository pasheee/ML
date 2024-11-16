import json
import torch
from WordDataset import WordDataset
from tqdm import tqdm
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
import numpy as np


def read_json(filepath):
    try:
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item)
        return data
        
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing the file: {e}")
        return None


def train_model(model, criterion, optimizer, dataloader, num_epoch, device = torch.device('cuda')):
    model.train()
    losses = []
    for epoch in range(1, num_epoch+1):
        print(f'epoch:{epoch}')
        for source, target in tqdm(dataloader):
            optimizer.zero_grad()
            
            target_input = target[:, :-1].to(device)
            target_output = target[:, 1:].to(device).flatten(start_dim = 0, end_dim = 1)

            outp = model(source.to(device), target_input).squeeze()
            outp = outp.flatten(start_dim = 0, end_dim = 1)

            
            loss = criterion(outp.to(device), target_output)
            loss.backward()
            optimizer.step()

            
            losses.append(loss.item())
    
    return losses



def make_wordinddicts(data, tokenizer):
    source = []
    target = []
    
    for line in data:
        t, s = line.keys()
        target.append(line[t].lower())
        source.append(line[s].lower())
    
    target_bag_of_words = []
    source_bag_of_words = []
    
    target_sentences = []
    source_sentences = []
    
    for i in range(len(target)):
        t_sent = target[i]
        s_sent = source[i]
        t_tokens = tokenizer.tokenize(t_sent.lower())
        s_tokens = [char for char in s_sent]
        
        target_bag_of_words.extend(t_tokens)
        source_bag_of_words.extend(s_tokens)
    
        target_sentences.append(t_tokens)
        source_sentences.append(s_tokens)
        
    
    special_symbols = ['<SOS>', '<EOS>', '<PAD>', '<UNK>']
    
    target_bag_of_words.extend(special_symbols)
    source_bag_of_words.extend(special_symbols)
    target_bag_of_words = set(target_bag_of_words)
    source_bag_of_words = set(source_bag_of_words)
    
    source_word2ind = {word: ind for ind, word in enumerate(source_bag_of_words)}
    target_word2ind = {word: ind for ind, word in enumerate(target_bag_of_words)}
    source_ind2word = {ind: word for ind, word in enumerate(source_bag_of_words)}
    target_ind2word = {ind: word for ind, word in enumerate(target_bag_of_words)}

    max_len = max(max([len(sentence) for sentence in target_sentences]), max([len(sentence) for sentence in source_sentences]))

    dataset = WordDataset(source_sentences, target_sentences, source_word2ind, target_word2ind, max_len = max_len)
    
    return source_word2ind, source_ind2word, target_word2ind, target_ind2word, max_len, dataset



def translate_sentence(model, sentence, source_word2ind, source_ind2word, target_word2ind, target_ind2word, device='cuda', max_length=5, temperature = 1):
    model.eval()

    source_tokens = sentence
        
    # Добавляем размерность батча
    source_tensor = torch.LongTensor([[source_word2ind['<SOS>']]+[source_word2ind.get(word, source_word2ind['<UNK>']) for word in source_tokens]+[source_word2ind['<EOS>']]]).to(device)
    # source_tensor = torch.LongTensor([[source_word2ind[word] for word in source_tokens]]).to(device)
    target_tokens = [target_word2ind['<SOS>']]
    
    with torch.no_grad():
        source_embeddings = model.source_embeddings(source_tensor)  
        _, encoded_hidden = model.encoder(source_embeddings) 
        
        for _ in range(max_length):
            target_tensor = torch.LongTensor([target_tokens]).to(device) 
            
            target_embeddings = model.target_embeddings(target_tensor)  
            
            output, _ = model.decoder(target_embeddings, encoded_hidden)
            output = model.non_lin(model.linear(model.non_lin(output)))
            logits = model.projection(output)/temperature
            
            probabilities = F.softmax(logits[0, -1], dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1).item()
            target_tokens.append(next_token)
            
            if next_token == target_word2ind['<EOS>']:
                break
    
    translated_tokens = target_tokens[1:-1] 
    translated_sentence = " ".join(target_ind2word[idx] for idx in translated_tokens)
    
    return translated_sentence



def evaluation(model, tokenizer, max_len, source_sentences, target_sentences, source_word2ind, source_ind2word, target_word2ind, target_ind2word, temperature = 0.2):
    bleu_scores = []
    
    for idx in tqdm(range(len(source_sentences))):
        source = source_sentences[idx]
        target = target_sentences[idx]  # Assuming target_sentences is a list of reference translations
        translated_sentence = translate_sentence(model, source, source_word2ind, source_ind2word, target_word2ind, target_ind2word, max_length=max_len, temperature=temperature)
    
        # Ensure the translated sentence is a list of tokens
        translated_sentence = translated_sentence.split() 
    
        # Calculate BLEU score.  The target needs to be a list of lists of references
        # If you only have one reference translation per source, make it a list of lists:
        score = sentence_bleu([target], translated_sentence, weights=(1/3, 1/3, 1/3, 0)) # target should be tokenized as well
        bleu_scores.append(score)
    
    average_bleu = np.mean(bleu_scores)
    print(f"Average BLEU score: {average_bleu}")



def write_json(data):
    with open('output.jsonl', 'w', encoding='utf-8') as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')

