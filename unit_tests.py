import os
import sys
import numpy as np
import torch 
import torch.nn as nn
from transformers import XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification, XLMRobertaForQuestionAnswering

def multiple_passes():
    tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-large')
    model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-large", num_labels=2)
    input = 'I am happy'
    source = tokenizer.encode_plus(input, return_tensors='pt')
    print(source)
    ids = source['input_ids']
    attention_mask = source['attention_mask']
    
    model.train()
    output = model(input_ids=ids, attention_mask=attention_mask)
    output_2 = model(input_ids=ids, attention_mask=attention_mask)

    print(output.logits)
    print(output_2.logits)

def weighted_dropout():
    pass

def main(args):
    
    if args[1] == 'multi':
        multiple_passes()
    elif args[1] == 'wd':
        weighted_dropout()

if __name__ == "__main__":
    args = sys.argv
    main(args)
