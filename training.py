import torch
import time
import math
import torch.nn as nn
import numpy as np
from transformers import *
from dataset import *
from config import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
torch.manual_seed(1337)

def evaluation(model, data_loader, device, num_batches = None):
    total_loss = 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(len(data_loader), num_batches)
    
    with torch.no_grad():
        for i , (input, target) in enumerate(data_loader):
            input, target = input.to(device), target.to(device)
            if i < num_batches:
                logits = model(input)
                total_loss += F.cross_entropy(logits.flatten(0, 1), target.flatten()).item()
            else:
                break
    return total_loss / num_batches

def generate(model, sample_text, max_text_length,tokenizer,config,  scale = 0.6667):
    encode = torch.tensor(tokenizer.encode(sample_text)).unsqueeze(0) # (1 , N)
    for i in range(max_text_length):
        ids_cond = encode[:, -config['context_length']:]
        with torch.no_grad():
            logits = model(ids_cond)
        last = logits[:, -1, :]
        
        # temperature scaling
        if scale is None:
            proba = torch.softmax(last, dim = -1)
        else:
            proba = torch.softmax(last / scale, dim = -1)
        
        next_ids = torch.multinomial(proba, num_samples=1)
        encode = torch.cat((encode, next_ids), dim = -1)
    new_text = tokenizer.decode(encode[0].tolist())
    return new_text


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_d = config_data()
    tokenizer = get_tokenizer() 
    # tokenizer = get_simple_tokenizer()
    config_m = config_model(tokenizer)
    train_loader, val_loader = get_dataset(tokenizer, config_d)
    print(f'Loading config and data...')
    print(f'Done loading config and data\n')
    print(config_d)

    GPT = GPT1(config_m)
    GPT = GPT.to(device)
    print(f'Total number of parameters: {sum(p.numel() for p in GPT.parameters()):,}')

    # parameters
    num_epochs = 5
    global_freq = -1
    eval_step = 10
    print(f'eval step: {eval_step}')

    # train
    optimizer = torch.optim.Adam(GPT.parameters(), lr = 3e-3)
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        start_time = time.time()
        GPT.train()
        for input, target in train_loader:
            input , target = input.to(device), target.to(device)
            optimizer.zero_grad()
            
            # loss
            logits = GPT(input)
            loss = F.cross_entropy(logits.flatten(0, 1), target.flatten())
            loss.backward()
            optimizer.step()
            global_freq += 1

            # evaluation
            if global_freq % eval_step == 0:
                GPT.eval()
                train_loss = evaluation(GPT, train_loader, device, num_batches = 5)
                val_loss   = evaluation(GPT, val_loader, device, num_batches = 5)
                train_losses.append(train_loss); val_losses.append(val_loss)
                GPT.train()

                print(f'freq {global_freq} | t_loss : {train_loss:.4f} | v_loss : {val_loss:.4f} | t : {time.time() - start_time:.3f}s')

            # stopping training if step > 2500
            if global_freq > 2500:
                break
        break
    print('Training is done.')

    # plotting
    plt.figure(figsize=(10,5))
    plt.plot(np.arange(0, global_freq, eval_step), train_losses, label = 'train loss' )
    plt.plot(np.arange(0, global_freq, eval_step), val_losses, label = 'val loss', ls = '--' )
    plt.xlabel("steps")
    plt.ylabel("loss"); plt.title("train/val loss")
    plt.legend()
    plt.show()

    # generate a new sentence
    GPT.eval()
    sample_text = "Hold my hand"
    new = generate(model = GPT, sample_text = sample_text, max_text_length = 300,tokenizer = tokenizer, config = config_m)
    print(f'generated text:\n{new}')

main()

