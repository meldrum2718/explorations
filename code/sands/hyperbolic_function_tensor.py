import torch
import numpy as np
from typing import Optional, Callable


class HyperbolicFunctionTensor:
    """
    Hyperbolic function representation using PyTorch.
    
    Stores function values on a regular meshgrid over the unit ball [-1,1]^n,
    with arbitrary coordinate transformations via pullback/pushforward functions.
    
    The hyperbolic mapping uses:
    - Ball to flat: atanh(||x||) * (x/||x||)  
    - Flat to ball: tanh(||x||) * (x/||x||)
    """
    
    def __init__(self, 
                 resolution: int, 
                 n_dims: int = 2, 
                 n_channels: int = 1,
                 pullback_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 pushforward_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 device: str = 'cpu'):
        """
        Initialize hyperbolic function tensor.
        
        Args:
            resolution: Grid resolution per dimension
            n_dims: Spatial dimensions  
            n_channels: Number of function channels
            pullback_fn: Global to local coordinate transformation
            pushforward_fn: Local to global coordinate transformation  
            device: PyTorch device ('cpu', 'cuda', 'mps')
        """
        self.resolution = resolution
        self.n_dims = n_dims
        self.n_channels = n_channels
        self.device = device
        
        # Default to identity transformations
        self.pullback_fn = pullback_fn or (lambda x: x)
        self.pushforward_fn = pushforward_fn or (lambda x: x)
        
        # Create ball coordinate meshgrid and value storage
        self.mesh_ball = self._create_meshgrid()
        self.values = torch.zeros(*self.mesh_ball.shape[:-1], n_channels, device=device)
    
    def _create_meshgrid(self) -> torch.Tensor:
        """Create n-dimensional meshgrid over [-1, 1]^n."""
        coords = torch.linspace(-1, 1, self.resolution, device=self.device, dtype=torch.float32)
        coord_arrays = torch.meshgrid(*[coords] * self.n_dims, indexing='ij')
        return torch.stack(coord_arrays, dim=-1)
    
    def b2f(self, x: torch.Tensor) -> torch.Tensor:
        """Ball to flat hyperbolic mapping: atanh(||x||) * (x/||x||)"""
        tol = 1e-18
        r = torch.linalg.norm(x, dim=-1, keepdim=True)
        r = torch.clamp(r, min=tol, max=1-tol)  # Avoid singularities
        
        unit_x = x / r
        return torch.arctanh(r) * unit_x
    
    def f2b(self, x: torch.Tensor) -> torch.Tensor:  
        """Flat to ball hyperbolic mapping: tanh(||x||) * (x/||x||)"""
        tol = 1e-15
        r = torch.linalg.norm(x, dim=-1, keepdim=True)
        r = torch.clamp(r, min=tol)  # Avoid division by zero
        
        unit_x = x / r
        return torch.tanh(r) * unit_x
    
    def sample_at(self, points: torch.Tensor) -> torch.Tensor:
        """
        Sample function values at global coordinate points.
        
        Args:
            points: Global coordinates, shape (batch_size, n_dims)
            
        Returns:
            Function values, shape (batch_size, n_channels)
        """
        # Transform: global -> local flat -> ball -> interpolate
        local_flat = self.pullback_fn(points)
        local_ball = self.f2b(local_flat)
        return self._interpolate_nd(local_ball)
    
    def _interpolate_nd(self, query_points: torch.Tensor) -> torch.Tensor:
        """N-dimensional linear interpolation on the ball grid."""
        batch_size = query_points.shape[0]
        
        # Map from [-1, 1] to [0, resolution-1] grid coordinates
        grid_coords = (query_points + 1) * (self.resolution - 1) / 2
        grid_coords = torch.clamp(grid_coords, 0, self.resolution - 1)
        
        # Split into integer and fractional parts
        grid_floor = torch.floor(grid_coords).long()
        grid_frac = grid_coords - grid_floor.float()
        grid_floor = torch.clamp(grid_floor, 0, self.resolution - 2)
        
        # Interpolate over 2^n_dims hypercube corners
        n_corners = 2 ** self.n_dims
        result = torch.zeros(batch_size, self.n_channels, device=self.device)
        
        for corner_idx in range(n_corners):
            # Generate corner offset using binary representation
            corner_offset = torch.tensor([
                (corner_idx >> d) & 1 for d in range(self.n_dims)
            ], dtype=torch.long, device=self.device)
            
            corner_coords = grid_floor + corner_offset
            
            # Compute multilinear interpolation weight
            weight = torch.ones(batch_size, device=self.device)
            for d in range(self.n_dims):
                if corner_offset[d] == 0:
                    weight *= (1 - grid_frac[:, d])
                else:
                    weight *= grid_frac[:, d]
            
            # Sample at corner and accumulate
            corner_values = self._sample_at_indices(corner_coords)
            result += weight.unsqueeze(1) * corner_values
        
        return result
    
    def _sample_at_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Sample values at integer grid indices."""
        batch_size = indices.shape[0]
        
        # Convert n-d indices to flat indices efficiently
        flat_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        multiplier = 1
        for d in range(self.n_dims - 1, -1, -1):
            flat_indices += indices[:, d] * multiplier
            multiplier *= self.resolution
        
        # Index into flattened values
        values_flat = self.values.reshape(-1, self.n_channels)
        return values_flat[flat_indices]
    
    @classmethod
    def from_function(cls, 
                     func: Callable[[torch.Tensor], torch.Tensor],
                     resolution: int,
                     n_dims: int = 2,
                     n_channels: int = 1,
                     pullback_fn: Optional[Callable] = None,
                     pushforward_fn: Optional[Callable] = None,
                     device: str = 'cpu') -> 'HyperbolicFunctionTensor':
        """
        Create hyperbolic function by sampling a function R^n -> R^c.
        
        Args:
            func: Function to sample, takes (batch_size, n_dims) -> (batch_size, n_channels)
            resolution: Grid resolution
            n_dims: Spatial dimensions
            n_channels: Number of function channels
            pullback_fn: Global to local transformation
            pushforward_fn: Local to global transformation
            device: PyTorch device
            
        Returns:
            New HyperbolicFunctionTensor with sampled values
        """
        hft = cls(resolution, n_dims, n_channels, pullback_fn, pushforward_fn, device)
        
        # Sample function: ball -> flat -> global -> evaluate
        mesh_flat = hft.mesh_ball.reshape(-1, n_dims)
        local_flat = hft.b2f(mesh_flat)
        global_coords = hft.pushforward_fn(local_flat)
        
        # Evaluate and store function values
        values = func(global_coords)
        if isinstance(values, np.ndarray):
            values = torch.tensor(values, device=device, dtype=torch.float32)
        if values.dim() == 1:
            values = values.unsqueeze(1)
        
        hft.values = values.reshape(*hft.mesh_ball.shape[:-1], n_channels)
        return hft
    
    def render_2d(self, resolution: int = 256, extent: float = 3.0) -> torch.Tensor:
        """
        Render 2D function as image by sampling on regular grid.
        
        Args:
            resolution: Output image resolution
            extent: Coordinate range [-extent, extent]
            
        Returns:
            Rendered image tensor, shape (resolution, resolution) or (resolution, resolution, n_channels)
        """
        if self.n_dims != 2:
            raise ValueError("render_2d only works for 2D functions")
        
        # Create sampling grid
        coords = torch.linspace(-extent, extent, resolution, device=self.device, dtype=torch.float32)
        X, Y = torch.meshgrid(coords, coords, indexing='ij')
        grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1)
        
        # Sample and reshape
        values = self.sample_at(grid_points)
        
        if self.n_channels == 1:
            return values[:, 0].reshape(resolution, resolution)
        else:
            return values.reshape(resolution, resolution, self.n_channels)
