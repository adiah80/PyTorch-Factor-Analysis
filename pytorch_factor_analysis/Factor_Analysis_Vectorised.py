import torch
import numpy as np
from torch import matmul as mul
from torch import inverse as inv
from tqdm import trange, tqdm

class FA_Vectorised:
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
            E1 = self._get_expectation1(X, L_pred, S_pred)
            E2 = self._get_expectation2(X, L_pred, S_pred)

            # MAXIMIZATION            
            L_new = self._get_new_L(X, E1, E2)
            S_new = self._get_new_S(X, E1, L_new)

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
        
    # -----------------------------------------------------
    # Internal Functions

    def _get_new_L(self, X, E1, E2):
        '''
        Returns prediction for L(n,d).
        '''
        L_pred = mul(mul(X,E1.T), inv(E2))
        return L_pred

    def _get_new_S(self, X, E1, L_new):
        '''
        Returns prediction for S(n,n).
        '''
        n = X.shape[0]
        S_pred = torch.diag(mul(X,X.T) - mul(L_new,mul(E1,X.T))) / n
        return S_pred

    def _get_expectation1(self, X, L, S):
        '''
        Return E1(d,m) from X(n,m), L(n,d) & S(n,n)
        Each column of E1 prepresents P(z|x) for a data sample.
        '''
        B = self._get_beta(L, S)
        E1 = mul(B, X)
        return E1

    def _get_expectation2(self, X, L, S):
        '''
        Returns E2(d,d) from X(n,m), L(n,d) & S(n,n)
        E2 represents sum(E(zz'|x)) where the sum is over all samples.
        '''
        d = L.shape[1]
        m = X.shape[1]
        I = torch.eye(d).to(self.device)
        B = self._get_beta(L, S)
        E2 = m*(I - mul(B,L)) + mul(mul(mul(B,X), X.T), B.T)
        return E2

    def _get_beta(self, L, S):
        '''
        Returns B(d,n) from L(n,d) and S(n,n)
        '''
        B = mul(L.T, inv(S + mul(L, L.T)))
        return B

