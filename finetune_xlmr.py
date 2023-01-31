import time
import torch
import argparse
import numpy as np
import evaluate
import datetime
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from transformers import XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification
from utils import data_to_df 

class CustomClassificationDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len
        self.input = self.data.src
        self.label = self.data.label

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        src = str(self.input[index])
        src = ' '.join(src.split())
        source = self.tokenizer.batch_encode_plus([src], max_length=self.max_len, padding="max_length", return_tensors='pt')
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        
        # list of numeric labels 
        tgt = self.label[index]

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'label': tgt
        }
 
def train(epoch, tokenizer, model, device, loader, optimizer, scheduler=None):
    model.train()
    start = time.time()
    loss_fn = nn.NLLLoss()
    for iteration, data in enumerate(loader, 0):
        x = data['source_ids'].to(device, dtype = torch.long)
        x_mask = data['source_mask'].to(device, dtype = torch.long)
        y = data['label'].to(device)
        
        output = model(input_ids = x, attention_mask = x_mask)
        y_hat = output.logits
        y_hat = F.log_softmax(y_hat, dim=1)
        loss = loss_fn(y_hat, y)
         
        if iteration%50 == 0:
            print(f'Epoch: {epoch}, Iteration: {iteration}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    end = time.time()
    print(f'Epoch: {epoch} used {end-start} seconds')

def validate(epoch, tokenizer, model, device, val_loader):
    
    model.eval()
    predictions = []
    actuals = []
    total_dev_loss = []
    loss_fn = nn.NLLLoss()
    with torch.no_grad():
        for index, data in enumerate(val_loader, 0):
            
            x = data['source_ids'].to(device, dtype = torch.long)
            x_mask = data['source_mask'].to(device, dtype = torch.long)
            y = data['label'].to(device, dtype = torch.long)

            output = model(input_ids = x, attention_mask = x_mask)
            y_hat = output.logits
            y_hat = F.log_softmax(y_hat, dim=1)
            loss = loss_fn(y_hat, y)
            
            predictions.extend(torch.argmax(y_hat, dim=1))
            actuals.extend(y)
            total_dev_loss.append(loss.item())
    
    
    #print(predictions[:5])
    #print(actuals[:5])
    #print(f'evaluation used {now-start} seconds')
    loss = np.mean(total_dev_loss)
    return predictions, actuals, loss

def main(args):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_class = 2 if args.task == 'pawsx' else 3
    tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')
    model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=n_class)
    model.to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr) 
    acc = evaluate.load("accuracy")
    loader_params = {
        'batch_size': args.bs,
        'shuffle': True,
        'num_workers': 0
    }
    
    train_dataset = data_to_df(task = args.task, language = 'en', split = 'train')
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    train_dataset = CustomClassificationDataset(train_dataset, tokenizer, args.max_len)
    train_loader = DataLoader(train_dataset, **loader_params)
    
    x = datetime.datetime.now()

    cols = str(x).strip().split()
    datestr = cols[0] + '-' + cols[1].split(':')[0]
    f = open(f'./logs/logs-{args.task}-{datestr}', 'a+')
    val_loaders = []

    #val_languages = ['en', 'de', 'es', 'bg', 'th', 'zh', 'ur', 'vi', 'ar', 'tr', 'fr', 'ru', 'hi', 'sw', 'el']
    val_languages = ['en', 'de', 'es', 'fr', 'ja', 'ko', 'zh']
    for language in val_languages:
        val_dataset = data_to_df(task = args.task, language = language, split = 'dev')
        val_dataset = CustomClassificationDataset(val_dataset, tokenizer, args.max_len)
        val_loaders.append(DataLoader(val_dataset, **loader_params))
    
    if args.epoch == 0:
        epoch = -1
        model.load_state_dict(torch.load(f"./{args.task}_latest.pth"))   
        print(f"loaded checkpoint at {args.task}_latest.pth")
        for index, val_loader in enumerate(val_loaders):
            total_acc = []
            y_hat, y, dev_loss = validate(epoch, tokenizer, model, device, val_loader)       
            result = acc.compute(references = y, predictions = y_hat)
            print(f"epoch = {epoch} | language = {val_languages[index]} | acc = {result['accuracy']}", file=f)
            print(f"epoch = {epoch} | language = {val_languages[index]} | acc = {result['accuracy']}")
            total_acc.append(result['accuracy'])
        print(f"total acc = {np.mean(total_acc)}")    
        print(f"total acc = {np.mean(total_acc)}", file=f)
                   
    for epoch in range(args.epoch):
        train(epoch, tokenizer, model, device, train_loader, optimizer)
        torch.save(model.state_dict(), f"{args.task}_latest.pth")
        
        for index, val_loader in enumerate(val_loaders):
            total_acc = []
            y_hat, y, dev_loss = validate(epoch, tokenizer, model, device, val_loader)       
            result = acc.compute(references = y, predictions = y_hat)
            print(f"epoch = {epoch} | language = {val_languages[index]} | acc = {result['accuracy']}", file=f)
            print(f"epoch = {epoch} | language = {val_languages[index]} | acc = {result['accuracy']}")
            total_acc.append(result['accuracy'])
            
        print(f"epoch = {epoch} | acc = {np.mean(total_acc)}", file=f)
        print(f"epoch = {epoch} | acc = {np.mean(total_acc)}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=15, type=int)
    parser.add_argument("--bs", default=32, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--task", default='xnli')
    parser.add_argument("--max_len", default=256, type=int)
    args = parser.parse_args()
    
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    
    main(args)
