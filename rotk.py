
'''
Author: Tianjian Li
Date: Feb 2, 2023

Code adapted from https://github.com/moskomule/ewc.pytorch/blob/master/utils.py
and
https://github.com/facebookresearch/fairseq/blob/main/examples/rxf/rxf_src/label_smoothed_cross_entropy_r3f.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable 
from copy import deepcopy

from transformers import XLMRobertaConfig 
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaEmbeddings

class Regularizer(object):
    
    def __init__(self, model, alpha, dataset, regularizer_type='l2'):
        self.model = model
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self.alpha = alpha
        self.dataset = dataset
        self.regularizer = regularizer_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if regularizer_type == 'ewc':
            self.fisher = self.compute_fisher()
        
        for n, p in deepcopy(self.params).items():
            self._means[n] = Variable(p.data)
    
    def compute_fisher(self):
        fisher = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            fisher[n] = Variable(p.data)

        self.model.eval()
        for data in self.dataset:
            x = data['source_ids'].to(self.device, dtype = torch.long)
            x_mask = data['source_mask'].to(self.device, dtype = torch.long)
            
            self.model.zero_grad()
            x = Variable(x)
            y = self.model(x, x_mask)
            y = y.logits.view(1, -1)
            label = y.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(y, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                fisher[n].data += p.grad.data ** 2 / len(self.dataset)

        fisher = {n: p for n, p in fisher.items()}
        return fisher

    def compute_sym_kl(self, noised_logits, input_logits):
        return (
            F.kl_div(F.log_softmax(noised_logits, dim=-1, dtype=torch.float32),
                    F.softmax(input_logits, dim=-1, dtype=torch.float32),
                    None,
                    None,
                    "sum",
                    ) +
            F.kl_div(F.softmax(input_logits, dim=-1, dtype=torch.float32),
                    F.log_softmax(noised_logits, dim=-1, dtype=torch.float32),
                    None,
                    None,
                    "sum",
                   ) 
              ) / noised_logits.size(0)
    
    
    def penalty(self, model, input_ids, attention_mask):

        loss = 0
        if self.regularizer == 'l2':
            for n, p in model.named_parameters():
                _loss = (p - self._means[n]) ** 2
                loss += _loss.sum()
            return self.alpha*loss

        elif self.regularizer == 'ewc':
            for n, p in model.named_parameters():
                _loss = self.fisher[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
            return self.alpha*loss
        
        elif self.regularizer == 'r3f':
            
            config = XLMRobertaConfig() 
            embedder = XLMRobertaEmbeddings(config)

            noise_sampler = torch.distributions.normal.Normal(loc=0.0, scale=self.eps)
            noise = noise_sampler.sample(sample_shape=token_embeddings.shape).to(token_embeddings)
            
            # feed to the model
            clean_embeddings = embedder(input_ids=input_ids)
            noised_embeddings = clean_embeddings.clone() + noise

            clean_logits = model(attention_mask=attention_mask, inputs_embeds=clean_embeddings)
            noised_logits = model(attention_mask=attention_mask, inputs_embeds=noised_embeddings)
            
            clean_logits = clean_logits.logits
            noised_logits = noised_logits.logits

            symm_kl = self.compute_sym_kl(noised_logits, clean_logits)

            return self.alpha * symm_kl
