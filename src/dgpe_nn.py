import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.notebook import tqdm
from torch.func import jvp

class DGPEModule(nn.Module):
    def __init__(self, dgpe, input_dim, output_dim, n_hidden=128):
        super(DGPEModule, self).__init__()
        self.dgpe = dgpe
        self.register_buffer('beta', torch.tensor(self.dgpe.beta, dtype=torch.float32))
        
        self.input_dim = input_dim
        self.output_dim = output_dim

        # self.sin_encoder = nn.Sequential(nn.Linear(input_dim, n_hidden), #nn.Dropout(p=0.2),
        #               nn.Linear(n_hidden, n_hidden),# nn.Sigmoid(),# nn.Dropout(p=0.2), nn.ReLU(), 
        #               nn.Linear(n_hidden, n_hidden)).to(torch.float32)
        
        # self.sin_decoder = nn.Sequential(nn.Linear(n_hidden, n_hidden),# nn.Dropout(p=0.2),nn.ReLU(),  
        #               nn.Linear(n_hidden, n_hidden), #nn.Sigmoid(),#nn.Dropout(p=0.2),nn.ReLU(), 
        #               nn.Linear(n_hidden, output_dim)).to(torch.float32)
        
        # self.cos_encoder = nn.Sequential(nn.Linear(input_dim, n_hidden),#nn.Dropout(p=0.2),
        #               nn.Linear(n_hidden, n_hidden),# nn.Sigmoid(),#nn.Dropout(p=0.2), nn.ReLU(), 
        #               nn.Linear(n_hidden, n_hidden)).to(torch.float32)
        
        # self.cos_decoder = nn.Sequential(nn.Linear(n_hidden, n_hidden),# nn.Dropout(p=0.2),nn.ReLU(),  
        #               nn.Linear(n_hidden, n_hidden),# nn.Sigmoid(),#nn.Dropout(p=0.2),nn.ReLU(), 
        #               nn.Linear(n_hidden, output_dim)).to(torch.float32)
        
        # self.lin_encoder = nn.Sequential(nn.Linear(input_dim, n_hidden),#nn.Dropout(p=0.2),
        #               nn.Linear(n_hidden, n_hidden),# nn.Sigmoid(),#nn.Dropout(p=0.2), nn.ReLU(), 
        #               nn.Linear(n_hidden, n_hidden)).to(torch.float32)
        
        # self.lin_decoder = nn.Sequential(nn.Linear(n_hidden, n_hidden), #nn.Dropout(p=0.2),nn.ReLU(),  
        #               nn.Linear(n_hidden, n_hidden), #nn.Sigmoid(),#nn.Dropout(p=0.2),nn.ReLU(), 
        #               nn.Linear(n_hidden, output_dim)).to(torch.float32)

        self.sin_encoder = nn.Sequential(nn.Linear(input_dim, n_hidden),
                      nn.LayerNorm(n_hidden), #nn.Dropout(p=0.2),
                      nn.Linear(n_hidden, n_hidden),# nn.Sigmoid(),# nn.Dropout(p=0.2), nn.ReLU(), 
                     nn.Linear(n_hidden, n_hidden)).to(torch.float32)
        
        self.sin_decoder = nn.Sequential(nn.Linear(n_hidden, n_hidden),
                                         nn.LayerNorm(n_hidden),# nn.Dropout(p=0.2),nn.ReLU(),  
                      nn.Linear(n_hidden, n_hidden),#nn.Sigmoid(),#nn.Dropout(p=0.2),nn.ReLU(), 
                      nn.Linear(n_hidden, output_dim)).to(torch.float32)
        
        self.cos_encoder = nn.Sequential(nn.Linear(input_dim, n_hidden), nn.LayerNorm(n_hidden),#nn.Dropout(p=0.2),
                      nn.Linear(n_hidden, n_hidden), # nn.Sigmoid(),#nn.Dropout(p=0.2), nn.ReLU(), 
                      nn.Linear(n_hidden, n_hidden)).to(torch.float32)
        
        self.cos_decoder = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.LayerNorm(n_hidden),# nn.Dropout(p=0.2),nn.ReLU(),  
                      nn.Linear(n_hidden, n_hidden), # nn.Sigmoid(),#nn.Dropout(p=0.2),nn.ReLU(), 
                      nn.Linear(n_hidden, output_dim)).to(torch.float32)
        
        self.lin_encoder = nn.Sequential(nn.Linear(input_dim, n_hidden), nn.LayerNorm(n_hidden),#nn.Dropout(p=0.2),
                      nn.Linear(n_hidden, n_hidden), # nn.Sigmoid(),#nn.Dropout(p=0.2), nn.ReLU(), 
                      nn.Linear(n_hidden, n_hidden)).to(torch.float32)
        
        self.lin_decoder = nn.Sequential(nn.Linear(n_hidden, n_hidden),  nn.LayerNorm(n_hidden),#nn.Dropout(p=0.2),nn.ReLU(),  
                      nn.Linear(n_hidden, n_hidden),#nn.Sigmoid(),#nn.Dropout(p=0.2),nn.ReLU(), 
                      nn.Linear(n_hidden, output_dim)).to(torch.float32)

        
        self.register_buffer('nn_idx_1', torch.tensor(self.dgpe.nn_idx_1, dtype=torch.int64).squeeze(0))
        self.register_buffer('nn_idx_2', torch.tensor(self.dgpe.nn_idx_2, dtype=torch.int64).squeeze(0))
        self.register_buffer('nn_idy_1', torch.tensor(self.dgpe.nn_idy_1, dtype=torch.int64).squeeze(0))
        self.register_buffer('nn_idy_2', torch.tensor(self.dgpe.nn_idy_2, dtype=torch.int64).squeeze(0))
        self.register_buffer('nn_idz_1', torch.tensor(self.dgpe.nn_idz_1, dtype=torch.int64).squeeze(0))
        self.register_buffer('nn_idz_2', torch.tensor(self.dgpe.nn_idz_2, dtype=torch.int64).squeeze(0))
        self.V = len(self.nn_idx_1)
        self.register_buffer('nn_idx_1_full', torch.cat([self.nn_idx_1, self.V + self.nn_idx_1], axis=0).squeeze(0))
        self.register_buffer('nn_idx_2_full', torch.cat([self.nn_idx_2, self.V + self.nn_idx_2], axis=0).squeeze(0))
        self.register_buffer('nn_idy_1_full', torch.cat([self.nn_idy_1, self.V + self.nn_idy_1], axis=0).squeeze(0))
        self.register_buffer('nn_idy_2_full', torch.cat([self.nn_idy_2, self.V + self.nn_idy_2], axis=0).squeeze(0))
        self.register_buffer('nn_idz_1_full', torch.cat([self.nn_idz_1, self.V + self.nn_idz_1], axis=0).squeeze(0))
        self.register_buffer('nn_idz_2_full', torch.cat([self.nn_idz_2, self.V + self.nn_idz_2], axis=0).squeeze(0))
        
    def dpsi_dt_fn(self, y, axis=1):
        dpsi_dt_fn = torch.cat([
            - (
                    # torch.gather(y[:,V:], axis, self.nn_idx_1) +
                    # torch.gather(y[:,V:], axis, self.nn_idx_2) +                    
                    # torch.gather(y[:,V:], axis, self.nn_idy_1) +
                    # torch.gather(y[:,V:], axis, self.nn_idy_2) +
                    # torch.gather(y[:,V:], axis, self.nn_idz_1) +
                    # torch.gather(y[:,V:], axis, self.nn_idz_2)
                    y[:,self.V:][:,self.nn_idx_1] +
                    y[:,self.V:][:,self.nn_idx_2] +                    
                    y[:,self.V:][:,self.nn_idy_1] +
                    y[:,self.V:][:,self.nn_idy_2] +
                    y[:,self.V:][:,self.nn_idz_1] +
                    y[:,self.V:][:,self.nn_idz_2]
                                        
            ) +
            self.beta * (torch.pow(y[:,self.V:], 2) + torch.pow(y[:,:self.V], 2)) * y[:,self.V:],
            (
                    # torch.gather(y[:,:V], axis, self.nn_idx_1) +
                    # torch.gather(y[:,:V], axis, self.nn_idx_2) +
                    # torch.gather(y[:,:V], axis, self.nn_idy_1) +
                    # torch.gather(y[:,:V], axis, self.nn_idy_2) +
                    # torch.gather(y[:,:V], axis, self.nn_idz_1) +
                    # torch.gather(y[:,:V], axis, self.nn_idz_2)
                    y[:,:self.V][:,self.nn_idx_1] +
                    y[:,:self.V][:,self.nn_idx_2] +                    
                    y[:,:self.V][:,self.nn_idy_1] +
                    y[:,:self.V][:,self.nn_idy_2] +
                    y[:,:self.V][:,self.nn_idz_1] +
                    y[:,:self.V][:,self.nn_idz_2]
            ) - self.beta * (torch.pow(y[:,self.V:], 2) + torch.pow(y[:,:self.V], 2)) * y[:,:self.V]
            ], dim=axis)#.squeeze(0)
        
        # max_norm = 1000.0  # Define a threshold
        # torch.nn.utils.clip_grad_norm_(dpsi_dt_fn, max_norm)
        return dpsi_dt_fn
    
    def dpsi_dt(self, t):
        # value, grad = jvp(self.forward, (t,), (t,))
        # Only differentiate w.r.t. the first column (time), not the latent z0
        tangent = torch.zeros_like(t)
        tangent[:, 0] = 1.0
        value, grad = jvp(self.forward, (t,), (tangent,))
        return grad
        
        # t = t.clone().requires_grad_(True)
        # J = torch.autograd.functional.jacobian(lambda x: self.forward(x), t, create_graph=True)
        # B = t.shape[0]
        # device = t.device    
        # idx = torch.arange(B, device=device)
        # dpsi_dt = J[idx, :, idx, 0]
        # # max_norm = 10.0  # Define a threshold
        # # torch.nn.utils.clip_grad_norm_(dpsi_dt, max_norm)
        # return dpsi_dt
    def E(self, y, axis=1):
        E = torch.sum( - y[:,self.V:] * (
                    # torch.gather(y[:,V:], axis, self.nn_idx_1) +
                    # torch.gather(y[:,V:], axis, self.nn_idx_2) +                    
                    # torch.gather(y[:,V:], axis, self.nn_idy_1) +
                    # torch.gather(y[:,V:], axis, self.nn_idy_2) +
                    # torch.gather(y[:,V:], axis, self.nn_idz_1) +
                    # torch.gather(y[:,V:], axis, self.nn_idz_2)
                    y[:,self.V:][:,self.nn_idx_1] +
                    y[:,self.V:][:,self.nn_idx_2] +                    
                    y[:,self.V:][:,self.nn_idy_1] +
                    y[:,self.V:][:,self.nn_idy_2] +
                    y[:,self.V:][:,self.nn_idz_1] +
                    y[:,self.V:][:,self.nn_idz_2]
                                        
            ) - y[:,:self.V] * (
                    # torch.gather(y[:,:V], axis, self.nn_idx_1) +
                    # torch.gather(y[:,:V], axis, self.nn_idx_2) +                    
                    # torch.gather(y[:,:V], axis, self.nn_idy_1) +
                    # torch.gather(y[:,:V], axis, self.nn_idy_2) +
                    # torch.gather(y[:,:V], axis, self.nn_idz_1) +
                    # torch.gather(y[:,:V], axis, self.nn_idz_2)
                    y[:,:self.V][:,self.nn_idx_1] +
                    y[:,:self.V][:,self.nn_idx_2] +                    
                    y[:,:self.V][:,self.nn_idy_1] +
                    y[:,:self.V][:,self.nn_idy_2] +
                    y[:,:self.V][:,self.nn_idz_1] +
                    y[:,:self.V][:,self.nn_idz_2]
            ) +
            self.beta * 0.5 * torch.pow(torch.pow(y[:,self.V:], 2) + torch.pow(y[:,:self.V], 2), 2), dim=axis)
        return E
    
    def N(self, y, axis=1):
        N = torch.sum(torch.pow(y, 2), dim=axis)
        return N
    
    def pinn_eq(self, t, y):
        return self.dpsi_dt(t) - self.dpsi_dt_fn(y)

    def forward(self, x):
        return (self.sin_decoder(torch.sin(self.sin_encoder(x))) + self.cos_decoder(torch.cos(self.cos_encoder(x))) + self.lin_decoder(self.lin_encoder(x)) + 
                self.sin_decoder(torch.sin(self.cos_encoder(x))) + self.cos_decoder(torch.cos(self.sin_encoder(x))))    
    

class _CircConv3d(nn.Module):
    """Conv3d with explicit circular padding — avoids PyTorch's gradient-scaling
    side-effect of padding_mode='circular' which can destabilise LBFGS."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=0).to(torch.float32)

    def forward(self, x):
        x = F.pad(x, [1, 1, 1, 1, 1, 1], mode='circular')
        return self.conv(x)


class UNetDGPEModule(nn.Module):
    def __init__(self, dgpe, input_shape, n_hidden=128, n_levels=None):
        super(UNetDGPEModule, self).__init__()
        self.dgpe = dgpe
        self.register_buffer('beta', torch.tensor(self.dgpe.beta, dtype=torch.float32))

        self.input_shape = input_shape
        min_dim = min(input_shape)
        max_levels = int(min_dim).bit_length() - 1  # floor(log2(min_dim))
        self.n_levels = min(n_levels if n_levels is not None else 4, max_levels)

        ch = n_hidden
        self.input_proj = nn.Conv3d(3, ch, kernel_size=1).to(torch.float32)
        self.encoders = nn.ModuleList([self.encoder_block(ch) for _ in range(self.n_levels)])
        self.b = _CircConv3d(ch, ch)
        self.decoder_ups = nn.ModuleList([self.decoder_upsample(ch) for _ in range(self.n_levels)])
        self.decoder_convs = nn.ModuleList([self.decoder_conv(ch) for _ in range(self.n_levels)])
        # Final projection back to 2 channels (Re, Im of ψ)
        self.out_conv = _CircConv3d(ch, 2)
        
        self.register_buffer('nn_idx_1', torch.tensor(self.dgpe.nn_idx_1, dtype=torch.int64).squeeze(0))
        self.register_buffer('nn_idx_2', torch.tensor(self.dgpe.nn_idx_2, dtype=torch.int64).squeeze(0))
        self.register_buffer('nn_idy_1', torch.tensor(self.dgpe.nn_idy_1, dtype=torch.int64).squeeze(0))
        self.register_buffer('nn_idy_2', torch.tensor(self.dgpe.nn_idy_2, dtype=torch.int64).squeeze(0))
        self.register_buffer('nn_idz_1', torch.tensor(self.dgpe.nn_idz_1, dtype=torch.int64).squeeze(0))
        self.register_buffer('nn_idz_2', torch.tensor(self.dgpe.nn_idz_2, dtype=torch.int64).squeeze(0))
        self.V = len(self.nn_idx_1)
        self.register_buffer('nn_idx_1_full', torch.cat([self.nn_idx_1, self.V + self.nn_idx_1], axis=0).squeeze(0))
        self.register_buffer('nn_idx_2_full', torch.cat([self.nn_idx_2, self.V + self.nn_idx_2], axis=0).squeeze(0))
        self.register_buffer('nn_idy_1_full', torch.cat([self.nn_idy_1, self.V + self.nn_idy_1], axis=0).squeeze(0))
        self.register_buffer('nn_idy_2_full', torch.cat([self.nn_idy_2, self.V + self.nn_idy_2], axis=0).squeeze(0))
        self.register_buffer('nn_idz_1_full', torch.cat([self.nn_idz_1, self.V + self.nn_idz_1], axis=0).squeeze(0))
        self.register_buffer('nn_idz_2_full', torch.cat([self.nn_idz_2, self.V + self.nn_idz_2], axis=0).squeeze(0))
    
    def encoder_block(self, num_channels=2):
        return nn.Sequential(
            _CircConv3d(num_channels, num_channels),
            nn.ReLU(),
            _CircConv3d(num_channels, num_channels),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

    def decoder_upsample(self, num_channels=2):
        return nn.ConvTranspose3d(num_channels, num_channels, kernel_size=2, stride=2).to(torch.float32)

    def decoder_conv(self, num_channels=2):
        return nn.Sequential(
            _CircConv3d(num_channels * 2, num_channels),
            nn.ReLU(),
            _CircConv3d(num_channels, num_channels),
            nn.ReLU()
        )
    
    def dpsi_dt_fn(self, y, axis=1):
        y = y.reshape(y.shape[0], -1)
        dpsi_dt_fn = torch.cat([
            - (
                    y[:,self.V:][:,self.nn_idx_1] +
                    y[:,self.V:][:,self.nn_idx_2] +                    
                    y[:,self.V:][:,self.nn_idy_1] +
                    y[:,self.V:][:,self.nn_idy_2] +
                    y[:,self.V:][:,self.nn_idz_1] +
                    y[:,self.V:][:,self.nn_idz_2]
                                        
            ) +
            self.beta * (torch.pow(y[:,self.V:], 2) + torch.pow(y[:,:self.V], 2)) * y[:,self.V:],
            (
                    y[:,:self.V][:,self.nn_idx_1] +
                    y[:,:self.V][:,self.nn_idx_2] +                    
                    y[:,:self.V][:,self.nn_idy_1] +
                    y[:,:self.V][:,self.nn_idy_2] +
                    y[:,:self.V][:,self.nn_idz_1] +
                    y[:,:self.V][:,self.nn_idz_2]
            ) - self.beta * (torch.pow(y[:,self.V:], 2) + torch.pow(y[:,:self.V], 2)) * y[:,:self.V]
            ], dim=axis)#.squeeze(0)
        
        # max_norm = 1000.0  # Define a threshold
        # torch.nn.utils.clip_grad_norm_(dpsi_dt_fn, max_norm)
        return dpsi_dt_fn
    
    def dpsi_dt(self, x):
        # x shape: (B, 3, Nx, Ny, Nz); channel 2 is dt broadcast across space
        # tangent selects derivative w.r.t. dt only
        tangent = torch.zeros_like(x)
        tangent[:, 2] = 1.0
        value, grad = jvp(self.forward, (x,), (tangent,))
        return grad.reshape(grad.shape[0], -1)
        
    def E(self, y, axis=1):
        y = y.reshape(y.shape[0], -1)
        E = torch.sum( - y[:,self.V:] * (
                    y[:,self.V:][:,self.nn_idx_1] +
                    y[:,self.V:][:,self.nn_idx_2] +                    
                    y[:,self.V:][:,self.nn_idy_1] +
                    y[:,self.V:][:,self.nn_idy_2] +
                    y[:,self.V:][:,self.nn_idz_1] +
                    y[:,self.V:][:,self.nn_idz_2]
                                        
            ) - y[:,:self.V] * (
                    y[:,:self.V][:,self.nn_idx_1] +
                    y[:,:self.V][:,self.nn_idx_2] +                    
                    y[:,:self.V][:,self.nn_idy_1] +
                    y[:,:self.V][:,self.nn_idy_2] +
                    y[:,:self.V][:,self.nn_idz_1] +
                    y[:,:self.V][:,self.nn_idz_2]
            ) +
            self.beta * 0.5 * torch.pow(torch.pow(y[:,self.V:], 2) + torch.pow(y[:,:self.V], 2), 2), dim=axis)
        return E
    
    def N(self, y, axis=1):
        N = torch.sum(torch.pow(y.reshape(y.shape[0], -1), 2), dim=axis)
        return N
    
    def pinn_eq(self, t, y):
        return self.dpsi_dt(t) - self.dpsi_dt_fn(y)

    def forward(self, x):
        # x shape: (B, 2, Nx, Ny, Nz)
        x = x.to(torch.float32)
        x = self.input_proj(x)
        skips = [x]
        h = x
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)
        b = self.b(skips.pop())
        for up, conv in zip(self.decoder_ups, self.decoder_convs):
            b = conv(torch.cat([up(b), skips.pop()], dim=1))
        out = self.out_conv(b)
        # Flatten spatial dims so downstream code receives (B, 2*V).
        # DO NOT remove this reshape — E / N / dpsi_dt_fn all expect flat input.
        return out.reshape(out.shape[0], -1)