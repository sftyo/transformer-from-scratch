import os
os.chdir("C:/Users/setyo.tirta/Desktop/study/transformer-from-scratch")

import torch
import math
import torch.nn as nn
from transformers import *
from dataset import *
from config import *
import torch.nn.functional as F



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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_d = config_data()
    tokenizer = get_tokenizer()
    config_m = config_model(tokenizer)
    train_loader, val_loader = get_dataset(tokenizer, config_d)
    print(f'Loading config and data...\n')
    print(f'{config_m}')
    print(f'{config_d}')

    GPT = GPT1(config_m)
    GPT = GPT.to(device)
    print(f'Total number of parameters: {sum(p.numel() for p in GPT.parameters())}')

    # parameters
    num_epochs = 5
    global_freq = -1
    eval_step = 100

    # train
    optimizer = torch.optim.AdamW(GPT.parameters(), lr = 3e-3, weight_decay = 1e-3)
    for epoch in range(num_epochs):
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
            GPT.eval()
            if global_freq % eval_step == 0:
                train_loss = evaluation(GPT, train_loader, device, num_batches = 5)
                val_loss = evaluation(GPT, val_loader, device, num_batches = 5)
            
            print(f'epoch {epoch+1} | freq {global_freq} | t_loss : {train_loss:.4f} | v_loss : {val_loss:.4f}')
            GPT.train()


main()