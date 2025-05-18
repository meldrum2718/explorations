"""
Neural field implementation using MLP with positional encoding.
This provides a randomly initialized neural field to visualize with stereographic projections.
"""

## TODO refactor this code ..

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NeuralField:
    def __init__(self, seed=None, d_h=64, n_layers=3, L=6, output_dim=1, device='cpu'):
        """
        Initialize a neural field with randomly initialized weights.
        
        Parameters:
        -----------
        seed : int, optional
            Random seed for reproducibility
        d_h : int
            Hidden dimension size
        n_layers : int
            Number of hidden layers in the MLP
        L : int
            Positional encoding parameter (determines number of frequencies)
        output_dim : int
            Output dimension (1 for scalar field, 3 for RGB, etc.)
        device : str
            Device to run the model on ('cpu', 'cuda', 'mps')
        """
        self.device = device
        self.L = L
        self.output_dim = output_dim
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        # Calculate input dimension with positional encoding
        d_in = 2 * (2 * L + 1)  # 2 input coordinates with positional encoding
        
        # Create model
        self.model = MLP(d_in=d_in, d_h=d_h, n_layers=n_layers, 
                         output_dim=output_dim, L=L).to(device)
        
    def __call__(self, x, y):
        """
        Evaluate the neural field at the given coordinates.
        
        Parameters:
        -----------
        x, y : numpy.ndarray
            Grid coordinates
            
        Returns:
        --------
        numpy.ndarray
            Field values at the given coordinates
        """
        # Reshape inputs to 2D points
        shape = x.shape
        coords = np.stack([x.flatten(), y.flatten()], axis=-1)
        
        # Convert to torch tensor
        coords_torch = torch.tensor(coords, dtype=torch.float32, device=self.device)
        
        ## TODO what ?? this seems wack .. dont normalize the coords ..
        # Normalize coordinates to [0, 1] range
        # (assuming x and y are already in a sensible range like [-1, 1] or [-10, 10])
        x_min, x_max = coords_torch[:, 0].min(), coords_torch[:, 0].max()
        y_min, y_max = coords_torch[:, 1].min(), coords_torch[:, 1].max()
        
        if x_max > x_min and y_max > y_min:  # Avoid division by zero
            normalized_coords = torch.zeros_like(coords_torch)
            normalized_coords[:, 0] = (coords_torch[:, 0] - x_min) / (x_max - x_min)
            normalized_coords[:, 1] = (coords_torch[:, 1] - y_min) / (y_max - y_min)
        else:
            normalized_coords = coords_torch

        # Evaluate model
        with torch.no_grad():
            values = self.model(normalized_coords)
        
        # Convert to numpy and reshape to original dimensions
        values_np = values.cpu().numpy()
        
        # Reshape based on output dimensions
        if self.output_dim == 1:
            return values_np.reshape(shape)
        else:
            return values_np.reshape(shape + (self.output_dim,))


def pos_enc(coords: torch.Tensor, L: int = 10):
    """
    Apply positional encoding to input coordinates.
    
    Parameters:
    -----------
    coords : torch.Tensor
        Input coordinates, shape (B, 2)
    L : int
        Number of frequency bands
        
    Returns:
    --------
    torch.Tensor
        Positionally encoded coordinates
    """
    B = coords.shape[0]
    x, y = coords[:, 0], coords[:, 1]
    
    device = coords.device
    frequencies = 2 ** torch.arange(L, dtype=torch.float32, device=device) * torch.pi
    
    x_frequencies = torch.einsum('b,f -> bf', x, frequencies)
    y_frequencies = torch.einsum('b,f -> bf', y, frequencies)
    
    x_sin = torch.sin(x_frequencies)
    x_cos = torch.cos(x_frequencies)
    y_sin = torch.sin(y_frequencies)
    y_cos = torch.cos(y_frequencies)
    
    pe = torch.cat([coords, x_sin, x_cos, y_sin, y_cos], dim=-1)
    
    return pe


## TODO what is this code organization?? why pos enc in mlp not in the nf directly ..
class MLP(nn.Module):
    def __init__(self, d_in, d_h, n_layers, output_dim=1, L=10):
        """
        Simple MLP for neural field representation.
        
        Parameters:
        -----------
        d_in : int
            Input dimension (after positional encoding)
        d_h : int
            Hidden dimension
        n_layers : int
            Number of hidden layers
        output_dim : int
            Output dimension
        L : int
            Positional encoding parameter
        """
        super(MLP, self).__init__()
        
        # Initial layer
        layers = [nn.Linear(d_in, d_h), nn.ReLU()]
        
        # Hidden layers
        for _ in range(n_layers):
            layers.append(nn.Linear(d_h, d_h))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(d_h, output_dim))
        
        self.layers = nn.Sequential(*layers)
        self.L = L
        self.output_dim = output_dim
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input coordinates, shape (B, 2)
            
        Returns:
        --------
        torch.Tensor
            Output values
        """
        x = pos_enc(x, L=self.L)
        x = self.layers(x)
        
        # Apply activation based on output dimension
        if self.output_dim == 1:
            # For scalar field, use sigmoid for nice visualization
            x = torch.sigmoid(x)
        elif self.output_dim == 3:
            # For RGB field, use sigmoid to get values in [0, 1]
            x = torch.sigmoid(x)
        
        return x


def create_random_field(seed=42, output_dim=1, device=torch.device('cpu')):
    """
    Create multiple random neural fields with different seeds.
    
    Parameters:
    -----------
    num_fields : int
        Number of fields to create
    seed : int, optional
        Base seed for reproducibility
    output_dim : int
        Output dimension for each field
    device : str
        Device to run the models on
        
    Returns:
    --------
    list
        List of NeuralField instances
    """
    fields = []
    
    for i in range(num_fields):
        
        # Vary hyperparameters slightly for diversity
        d_h = np.random.randint(32, 128)
        n_layers = np.random.randint(2, 5)
        L = np.random.randint(4, 10)
        
        field = NeuralField(
            seed=seed,
            d_h=d_h,
            n_layers=n_layers,
            L=L,
            output_dim=output_dim,
            device=device
        )
        fields.append(field)
    
    return fields
