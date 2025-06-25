import torch
import torch.nn.functional as F
import networkx as nx

from function_tensor import FunctionTensor, normalize
from typing import List



def interlaced_perm(d):
    """
    Creates an interlaced permutation as a tuple of length d.
    
    For even d:
      - Even positions (0,2,4,...) get filled with (0,1,2,...,d//2-1)
      - Odd positions (1,3,5,...) get filled with (d//2,d//2+1,...,d-1)
    
    For odd d:
      - Even positions (0,2,4,...) get filled with (d//2,d//2+1,...,d-1)
      - Odd positions (1,3,5,...) get filled with (0,1,2,...,d//2-1)
    
    Args:
        d: Integer length of permutation
        
    Returns:
        tuple: Interlaced permutation indices
    """
    offset = d % 2  # 1 if odd, 0 if even
    first_half = list(range(d//2))
    second_half = list(range(d//2, d))
    indices = [0] * d
    
    # Positions determined by offset
    for i in range(offset, d, 2):
        indices[i] = first_half[i//2 - offset//2]
    
    for i in range(1-offset, d, 2):
        indices[i] = second_half[(i-(1-offset))//2]
        
    return tuple(indices)


def flatten(X: torch.Tensor):
    k = X.dim()
    perm = interlaced_perm(k)
    X = X.permute(perm)
    permuted_shape = torch.tensor(X.shape).to(int)
    new_shape = [
        torch.prod(permuted_shape[:(k//2)]).item(),
        torch.prod(permuted_shape[(k//2):]).item()
    ]

    X = X.reshape(new_shape)
    return X



class FunctionTensorNetwork:
    """
    Stores an nd tensor representing a function f: R^n -> R^c sampled on a regular grid.
    Treats the function as periodic with period 2 over [-1, 1]^n.
    """
    def __init__(self, fts: List[FunctionTensor]):
        """
        Args:
            tensor: shape (resolution, resolution, ..., resolution, C)
                   representing function samples on [0,1]^n grid
        """

        self.fts = fts
        self.n = len(fts)
        

    def step(self, adj, pushforwards, alpha):
        warped = []
        for ft, pushforward in zip(self.fts, pushforwards):
            mesh = FunctionTensor.generate_global_mesh_coords(ft.resolution, ft.n_dims)
            warped.append(ft.sample(pushforward(mesh)))
        outp  = torch.stack(warped, dim=0) # output from each ft
        inp = (adj @ outp.reshape(self.n, -1)).reshape(outp.shape)
        tensors = (1 - alpha) * self._get_tensors() + alpha * inp

        self._set_tensors(normalize(tensors))

    def _get_tensors(self):
        return torch.stack([ft.tensor for ft in self.fts], dim=0)

    def _set_tensors(self, tensors):
        for idx, tensor in enumerate(tensors):
            self.fts[idx].tensor = tensor

    def to_flat(self):
        return flatten(self._get_tensors())
    
    def __call__(self, x_global):
        """
        Evaluate function at global coordinates
        
        Args:
            x_global: (..., n_dims) points in [-1,1]^n space
            
        Returns:
            (..., channels) function values
        """
        # 1. Convert global to local coordinates
        x_local = FunctionTensor.global_to_local(x_global)
        
        # 2. Apply periodic boundary conditions
        x_wrapped = wrap_coordinates(x_local)
        
        # 3. Interpolate tensor values
        return self.nd_linear_interpolate(x_wrapped)

    def resample(self, resolution):
        mesh = FunctionTensor.generate_global_mesh_coords(resolution, self.n_dims)
        tensor = self(mesh)
        return FunctionTensor(tensor)


    @classmethod
    def global_to_local(cls, x_global):
        """Convert from global [-1,1]^n coords to local [0,1]^n coords"""
        return (x_global + 1.0) / 2.0
    
    @classmethod
    def local_to_global(cls, x_local):
        """Convert from local [0,1]^n coords to global [-1,1]^n coords"""
        return x_local * 2.0 - 1.0

    @classmethod
    def generate_local_mesh_coords(cls, resolution, n_dims):
        """ Generate regular mesh over [0, 1]^n """
        coords_1d = torch.linspace(0, 1, resolution)
        grids = torch.meshgrid([coords_1d] * n_dims, indexing='ij')
        local_coords = torch.stack(grids, dim=-1)
        return local_coords

    @classmethod
    def generate_global_mesh_coords(cls, resolution, n_dims):
        """ Generate regular mesh over [0, 1]^n """
        local_coords = FunctionTensor.generate_local_mesh_coords(resolution, n_dims)
        return FunctionTensor.local_to_global(local_coords)

    @classmethod
    def from_function(cls, func, resolution, n_dims, channels):
        """
        Create FunctionTensor by sampling a function on the grid
        
        Args:
            func: callable that takes (n_dims,) coords and returns (channels,) values
            resolution: grid resolution per dimension
            n_dims: number of spatial dimensions
            channels: number of output channels
        """
        global_coords = FunctionTensor.generate_global_mesh_coords(resolution, n_dims)
        values_tensor = func(global_coords.reshape(-1, n_dims)).reshape([resolution]*n_dims)
        if channels == 1:
            values_tensor = values_tensor.unsqueeze(-1)
        return FunctionTensor(values_tensor)
