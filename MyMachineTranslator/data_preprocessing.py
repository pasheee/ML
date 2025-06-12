from WordDataset import WordDataset

TARGET_MAX_LEN = 20
SOURCE_MAX_LEN = 20

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
        s_tokens = s_sent.split()
        
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

    target_max_len, source_max_len = max([len(sentence) for sentence in target_sentences]), max([len(sentence) for sentence in source_sentences])

    dataset = WordDataset(source_sentences, target_sentences, source_word2ind, target_word2ind, source_max_len = SOURCE_MAX_LEN, target_max_len = TARGET_MAX_LEN)

    return source_word2ind, source_ind2word, target_word2ind, target_ind2word, source_max_len, target_max_len, dataset

