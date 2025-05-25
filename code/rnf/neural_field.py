import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralField(nn.Module):
    def __init__(self, d_in, d_out=1, hidden_dim=256, n_layers=3, n_freqs=10):
        super().__init__()
        self.n_freqs = n_freqs
        self.freqs = 2. ** torch.arange(n_freqs, dtype=torch.float32)
        
        pos_dim = d_in * n_freqs * 2
        layers = [nn.Linear(pos_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, d_out))
        self.net = nn.Sequential(*layers)
    
    def positional_encoding(self, x):
        freqs = self.freqs.to(x.device)
        x_freq = x.unsqueeze(-1) * freqs
        return torch.cat([torch.sin(x_freq), torch.cos(x_freq)], dim=-1).flatten(-2)
    
    def forward(self, x):
        shape = x.shape[:-1]
        flat_x = x.reshape(-1, x.shape[-1])
        encoded = self.positional_encoding(flat_x)
        output = self.net(encoded)
        return output.reshape(*shape, -1)
