import torch
import numpy as np

def kd(consolidated, new, alpha):

    return alpha * kl_divergence(consolidated, new)

def l2_reg(consolidated, new, alpha):
    
    count = 0
    for p, p_prime in zip(consolidated, new):
        reg_term += (p - p_prime) ** 2
        count += 1               
    return alpha * reg_term / count

def ewc(consolidated, new, alpha, emp_fisher):
    
    count = 0
    for p, p_prime in zip(consolidated, new):
        reg_term += emp_fisher[p]*(p - p_prime) ** 2
        count += 1
    return alpha * reg_term / count 

