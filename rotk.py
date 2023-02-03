
'''
Author: Tianjian Li
Date: Feb 2, 2023

Code adapted from https://github.com/moskomule/ewc.pytorch/blob/master/utils.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable 

class Regularizer(object):
    def __init__(self, model, alpha, dataset, regularizer_type='l2'):
        self.model = model
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self.alpha = alpha
        self.dataset = dataset
        self.regularizer = regularizer_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if regularizer == 'ewc':
            self.fisher = self.compute_fisher(self)
        
        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)
    
    def compute_fisher(self):
        fisher = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            fisher[n] = variable(p.data)

        #self.model.eval()
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

    def penalty(self, model):

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
