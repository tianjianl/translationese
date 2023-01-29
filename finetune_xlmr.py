import torch
import numpy as np
import pandas as pd

from utils import data_preprocess 

from torch import cuda
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from transformers import XLMRobertaTokenizerFast, XLMRobertaModel

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len
        self.input = self.data.src
        self.prediction = self.data.tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        src = str(self.src[index])
        src = ' '.join(src.split())





def main(args):
    
    tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')
    model = XLMRobertaModel.from_pretrained("xlm-roberta-base")

    # prepare input
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer.batch_encode_plus(text, return_tensors='pt')

    # forward pass
    output = model(**encoded_input)
    pooled_output = torch.mean(output.last_hidden_state, dim=1)
    print(pooled_output.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--bs", default=32, type=int)
    args = parser.parse_args()
    
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    
    main(args)
