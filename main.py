import torch
from pytorch_factor_analysis import FA_Vectorised, FA_Numpy, FA
from pytorch_factor_analysis import generate_sample_data, plot
from configs import cfg1, cfg2, cfg3

def main():
    cfg = cfg3
    device = torch.device('cpu')       # Training is faster on CPU
    # device = torch.device('cuda')      # Uncomment for training on GPU

    Z, L, S, U, X = generate_sample_data(cfg, toTensor=True, device=device)
    fa = FA(cfg, device)
    L_pred, S_pred = fa.train_EM(X, L, S)
    plot(fa.metrics['L_error'])
    plot(fa.metrics['S_error'])

if __name__ == "__main__":
    main()

