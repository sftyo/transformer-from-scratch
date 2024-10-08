import re
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

class Data(Dataset):
    def __init__(self, text, tokenizer, config):
        self.input_id = []
        self.target_id = []
        encoder = tokenizer.encode(text)
        for i in range(0, len(encoder) - config['max_length'], config['stride']):
            input = encoder[i:i + config['max_length']]
            target = encoder[i + 1:i + config['max_length'] + 1]
            self.input_id.append(torch.tensor(input))
            self.target_id.append(torch.tensor(target))
    
    def __len__(self):
        return len(self.input_id)

    def __getitem__(self, idx):
        return self.input_id[idx], self.target_id[idx]

class SimpleTokenizer():
    def __init__(self, vocab):
        self.stoi = vocab
        self.itos = {i : s for s, i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.stoi else "<|unk|>" for item in preprocessed]
        ids = [self.stoi[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.itos[i]] for i in ids)
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
    
    def n_vocab(self):
        return len(self.stoi)       
    
def get_simple_tokenizer():
    files = "the-verdict.txt"
    with open(files, 'r', encoding='utf-8') as f:
        text = f.read()
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>" , "<|unk|>"])
    vocab = {i:s for s,i in enumerate(all_tokens)}
    
    tokenizer = SimpleTokenizer(vocab)
    return tokenizer
 
def get_tokenizer():
    # BPE tokenizer
    tokenizer = tiktoken.get_encoding('gpt2')
    return tokenizer

def get_dataset(tokenizer, config):
    files = "the-verdict.txt"
    with open(files, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # train / val split
    split = 0.9
    train = text[:int(split * len(text))]
    val = text[int(split * len(text)):]
    
    # data loader
    train_data = Data(train, tokenizer, config)
    val_data = Data(val, tokenizer, config)
    train_loader = DataLoader(train_data, batch_size = config['batch_size'],
                              shuffle = True, drop_last = True, num_workers = 0)
    val_loader = DataLoader(val_data, batch_size = config['batch_size'],
                            shuffle = True, drop_last = True, num_workers = 0)
    
    return train_loader, val_loader