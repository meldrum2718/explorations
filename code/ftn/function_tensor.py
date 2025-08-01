import torch
import torch.nn.functional as F

from utils import mod1, normalize, flatten



class FunctionTensor:
    """
    Stores an nd tensor representing a function f: R^n -> R^c sampled on a regular grid.
    Treats the function as periodic with period 2 over [-1, 1]^n.
    """
    
    def __init__(self, tensor):
        """
        Args:
            tensor: shape (resolution, resolution, ..., resolution, C)
                   representing function samples on [0,1]^n grid
        """
        self.tensor = tensor
        self.shape = tensor.shape
        self.n_dims = len(self.shape) - 1  # spatial dimensions
        self.resolution = self.shape[0]  # assuming uniform resolution
        self.channels = self.shape[-1]
        
        # Validate tensor shape
        assert all(s == self.resolution for s in self.shape[:-1]), \
            "All spatial dimensions must have same resolution"
    
    
    def nd_linear_interpolate(self, x_local):
        """
        Custom n-dimensional linear interpolation
        
        Args:
            x_local: (..., n_dims) coordinates in [0,1]^n
            
        Returns:
            (..., channels) interpolated values
        """
        batch_shape = x_local.shape[:-1]  # (...,)
        n_points = x_local.numel() // self.n_dims
        x_flat = x_local.view(n_points, self.n_dims)
        
        # Apply proper periodic wrapping to ensure we're in [0,1)
        x_wrapped = mod1(x_flat)
        
        # Convert [0,1] coords to grid coords [0, resolution-1]
        grid_coords = x_wrapped * (self.resolution - 1)  # (n_points, n_dims)
        
        # Find floor and ceil indices for each dimension
        indices_low = torch.floor(grid_coords).long()  # (n_points, n_dims)
        indices_high = indices_low + 1
        
        # Handle wraparound at boundaries (periodicity)
        indices_high = torch.where(indices_high >= self.resolution, 0, indices_high)
        
        # Interpolation weights: distance from low index
        weights = grid_coords - indices_low.float()  # (n_points, n_dims)
        
        # Generate all 2^n corner combinations
        n_corners = 2 ** self.n_dims
        corner_offsets = torch.zeros(n_corners, self.n_dims, dtype=torch.long, device=x_local.device)
        
        for i in range(n_corners):
            for d in range(self.n_dims):
                corner_offsets[i, d] = (i >> d) & 1  # extract d-th bit
        
        # For each point and each corner, compute indices and weights
        interpolated = torch.zeros(n_points, self.channels, device=x_local.device, dtype=self.tensor.dtype)
        
        for corner_idx in range(n_corners):
            # Get indices for this corner
            corner_indices = torch.where(
                corner_offsets[corner_idx:corner_idx+1] == 0,  # (1, n_dims)
                indices_low,  # (n_points, n_dims) 
                indices_high  # (n_points, n_dims)
            )  # (n_points, n_dims)
            
            # Compute weight for this corner (product across dimensions)
            corner_weight = torch.ones(n_points, device=x_local.device)
            for d in range(self.n_dims):
                if corner_offsets[corner_idx, d] == 0:
                    corner_weight *= (1 - weights[:, d])  # weight toward low index
                else:
                    corner_weight *= weights[:, d]  # weight toward high index
            
            # Advanced indexing to get tensor values at corner positions
            indices_tuple = tuple(corner_indices[:, d] for d in range(self.n_dims))
            corner_values = self.tensor[indices_tuple]  # (n_points, channels)
            
            # Accumulate weighted contribution
            interpolated += corner_weight.unsqueeze(-1) * corner_values
        
        return interpolated.view(*batch_shape, self.channels)
    
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
        x_wrapped = mod1(x_local)
        
        # 3. Interpolate tensor values
        return self.nd_linear_interpolate(x_wrapped)

    def sample(self, x_global):
        return self.__call__(x_global)


    def resample(self, resolution):
        ## TODO name this better, eg resize
        mesh = FunctionTensor.generate_global_mesh_coords(resolution, self.n_dims)
        tensor = self(mesh)
        return FunctionTensor(tensor)

    def blend(self, new_state, blend_factor):
        new_state = (1 - blend_factor) * self.tensor + blend_factor * normalize(new_state)
        self.tensor = normalize(new_state)


    def to_flat(self):
        return flatten(self.tensor)


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
        """ Generate regular mesh over [-1, 1]^n """
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
