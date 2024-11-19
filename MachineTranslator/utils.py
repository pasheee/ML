import json
import torch
import torch.nn as nn
from WordDataset import WordDataset
from tqdm import tqdm
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
TARGET_MAX_LEN = 50
SOURCE_MAX_LEN = 200

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



def translate(model, sentence, source_word2ind, target_word2ind, beam_size=4, max_length=TARGET_MAX_LEN, device='cuda'):
    model.eval()
    source_ids = torch.tensor([[source_word2ind.get(word, source_word2ind['<UNK>']) for word in sentence]]).to(device)

    # Начальная точка
    beams = [(torch.tensor([[target_word2ind['<SOS>']]]).to(device), 0)]  # (sequence, score)
    completed = []

    with torch.no_grad():
        for _ in range(max_length):
            new_beams = []
            for seq, score in beams:
                output = model(source_ids, seq)
                probs = nn.functional.softmax(output[0, -1], dim=-1)
                topk = probs.topk(beam_size)

                for i in range(beam_size):
                    new_seq = torch.cat([seq, torch.tensor([[topk.indices[i]]]).to(device)], dim=1)
                    new_score = score + topk.values[i].item()
                    if topk.indices[i].item() == target_word2ind['<EOS>']:
                        completed.append((new_seq, new_score))
                    else:
                        new_beams.append((new_seq, new_score))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            if len(completed) >= beam_size:
                break

    best_seq = max(completed, key=lambda x: x[1])[0] if completed else beams[0][0]
    target_ind2word = {v: k for k, v in target_word2ind.items()}
    translated = [target_ind2word[idx.item()] for idx in best_seq[0][1:-1]]
    return ' '.join(translated)

def write_json(data):
    with open('output.jsonl', 'w', encoding='utf-8') as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')

