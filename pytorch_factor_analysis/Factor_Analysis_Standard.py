import torch
import numpy as np
from torch import matmul as mul
from torch import inverse as inv
from tqdm import trange, tqdm

class FA_Standard:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        
    def train_EM(self, X, true_L, true_S):        
        # Initialize lists for training metrics
        self.metrics = {
            'L_error': [],
            'S_error': [],
        }
        
        # Set random Values for Params
        L_pred = torch.Tensor(np.random.rand(self.cfg['NUM_FEATURES'], self.cfg['NUM_FACTORS'])).to(self.device)
        S_pred = torch.Tensor(np.diag(np.random.randn(self.cfg['NUM_FEATURES']))).to(self.device)

        # Run the EM Loop
        for iteration in trange(self.cfg['NUM_ITERATIONS']):

            # EXPECTATION
            E1_arr = []
            E2_arr = []
            for sample_idx in range(self.cfg['NUM_SAMPLES']):
                x = X[:,sample_idx]
                x = x.reshape(x.shape[0],-1)
                E1_arr.append(self._get_expectation1(x, L_pred, S_pred))
                E2_arr.append(self._get_expectation2(x, L_pred, S_pred))

            # MAXIMIZATION            
            L_new = self._get_new_L(X, E1_arr, E2_arr)
            S_new = self._get_new_S(X, E1_arr, L_new)

            # Check Convergence
            DIFF_THRESHOLD = 1e-6
            Delta_L = torch.abs(L_pred - L_new).mean()
            Delta_S = torch.abs(S_pred - S_new).mean()
            if(Delta_L < DIFF_THRESHOLD and Delta_S < DIFF_THRESHOLD):
                break

            L_pred = L_new
            S_pred = S_new

            # Log Errors
            self.metrics['L_error'].append(torch.abs(true_L - L_pred).mean())
            self.metrics['S_error'].append(torch.abs(true_S - S_pred).mean())

            if iteration % self.cfg['LOG_FREQ'] == 0:
                tqdm.write("[{:06d}/{:06d}] L_error: {:.10f} \t S_error: {:.10f} \r"\
                           .format(iteration, 
                                   self.cfg['NUM_ITERATIONS'], 
                                   self.metrics['L_error'][-1], 
                                   self.metrics['S_error'][-1]),
                          end="\n")
        print('------------------------------------')
        print('Training Done.')
        self.L_pred = L_pred
        self.S_pred = S_pred
        return L_pred, S_pred

    def _get_new_L(self, X, E1_arr, E2_arr):
        arr = []
        for i in range(X.shape[1]):
            x = X[:,i]
            x = x.reshape(x.shape[0],-1)
            arr.append(mul(x, E1_arr[i].T))
        t1 = torch.stack(arr).sum(0)
            
        t2 = inv(torch.stack(E2_arr).sum(axis=0))
        return mul(t1, t2)

    def _get_new_S(self, X, E1, L_new):
        arr = []
        for i in range(X.shape[1]):
            x = X[:,i]
            x = x.reshape(x.shape[0],-1)
            arr.append(mul(x,x.T) - mul(mul(L_new, E1[i]), x.T))
        t = torch.diag(torch.diag(torch.stack(arr).mean(0)))
        return t

    def _get_expectation1(self, x, L, S):
        B = self._get_beta(L, S)
        E1 = mul(B, x)
        return E1

    def _get_expectation2(self, x, L, S):
        I = torch.eye(L.shape[1]).to(self.device)
        B = self._get_beta(L, S)
        E2 = I - mul(B,L) + mul(mul(mul(B,x), x.T), B.T)
        return E2

    def _get_beta(self, L, S):
        B = mul(L.T, inv(S + mul(L, L.T)))
        return B