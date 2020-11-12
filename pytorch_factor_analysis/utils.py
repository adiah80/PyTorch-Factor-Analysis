import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_sample_data(cfg, toTensor, device):
    np.random.seed(cfg['RANDOM_SEED'])
    Z = np.random.normal(0, 1, [cfg['NUM_FACTORS'],cfg['NUM_SAMPLES']])  # ~ N(0,1)
    L = np.random.rand(cfg['NUM_FEATURES'], cfg['NUM_FACTORS'])
    S = np.abs(np.diag(np.random.rand(cfg['NUM_FEATURES'])))            # Is a diagonal matrix

    U = np.zeros([cfg['NUM_FEATURES'], cfg['NUM_SAMPLES']])
    for i in range(cfg['NUM_SAMPLES']):
        U[:,i] = np.random.normal(0, np.diag(S))       
    
    X = np.matmul(L,Z) + U
    if toTensor:
        Z = torch.Tensor(Z).to(device)
        L = torch.Tensor(L).to(device)
        S = torch.Tensor(S).to(device)
        U = torch.Tensor(U).to(device)
        X = torch.Tensor(X).to(device)
    
    return Z, L, S, U, X

def plot(values, metric):
    plt.figure(figsize=[16,9])
    plt.title(f"{metric} vs EM iterations.")
    plt.plot(values, label=metric)
    plt.xlabel("EM Iterations")
    plt.ylabel("Prediction Error")
    plt.legend()
    plt.grid()
    plt.show()