import torch
import numpy as np


def sample_unit_ball(n_samples, d, device='cpu'):
    """Sample points uniformly from unit ball interior"""
    x = torch.randn(n_samples, d, device=device)
    norms = torch.norm(x, dim=-1, keepdim=True)
    x = x / norms
    r = torch.rand(n_samples, 1, device=device) ** (1/d)
    return x * r


def generate_mesh_grid(resolution, d, domain_range=(-1.5, 1.5), device='cpu'):
    """Generate regular mesh grid points"""
    if isinstance(resolution, int):
        resolution = [resolution] * d
    
    # Create 1D grids for each dimension
    grids = []
    for i in range(d):
        if isinstance(domain_range, tuple):
            grid = torch.linspace(domain_range[0], domain_range[1], resolution[i], device=device)
        else:
            grid = torch.linspace(domain_range[i][0], domain_range[i][1], resolution[i], device=device)
        grids.append(grid)
    
    # Create meshgrid
    mesh_grids = torch.meshgrid(*grids, indexing='ij')
    coords = torch.stack(mesh_grids, dim=-1)
    
    # Flatten to (N, d) format
    flat_coords = coords.reshape(-1, d)
    
    return flat_coords


def filter_points_in_unit_ball(points):
    """Filter points to only include those inside unit ball"""
    norms = torch.norm(points, dim=-1)
    mask = norms <= 1.0
    return points[mask]


class Transformation:
    """Rotation + scaling transformation in 2D"""
    def __init__(self, angle=0.5, scale_range=(0.8, 1.2)):
        self.angle = angle
        self.scale_min, self.scale_max = scale_range
    
    def transform_points(self, x):
        """Transform input points"""
        if x.shape[-1] == 2:
            # Random scaling factor for each batch
            scale = torch.rand(x.shape[0], 1, device=x.device, dtype=x.dtype) * (self.scale_max - self.scale_min) + self.scale_min
            
            # Rotation matrix
            cos_a = torch.cos(torch.tensor(self.angle, device=x.device, dtype=x.dtype))
            sin_a = torch.sin(torch.tensor(self.angle, device=x.device, dtype=x.dtype))
            R = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], device=x.device, dtype=x.dtype)
            
            # Apply scaling then rotation
            x_scaled = x * scale
            x_transformed = x_scaled @ R.T
            
            return x_transformed, R, scale
        else:
            return x, None, None
    
    def transform_vectors(self, vectors, R=None, scale=None):
        """Transform vector field values (for equivariance)"""
        if R is not None and vectors.shape[-1] == 2:
            # Apply rotation to vectors
            vectors_rotated = vectors @ R.T
            if scale is not None:
                # Apply scaling
                vectors_scaled = vectors_rotated * scale
                return vectors_scaled
            return vectors_rotated
        return vectors
    
    def __call__(self, x):
        """For backward compatibility - just transform points"""
        transformed_x, _, _ = self.transform_points(x)
        return transformed_x


class RotationTransformation:
    """Pure rotation transformation"""
    def __init__(self, angle=0.5):
        self.angle = angle
    
    def transform_points(self, x):
        """Transform input points"""
        if x.shape[-1] == 2:
            cos_a = torch.cos(torch.tensor(self.angle, device=x.device, dtype=x.dtype))
            sin_a = torch.sin(torch.tensor(self.angle, device=x.device, dtype=x.dtype))
            R = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], device=x.device, dtype=x.dtype)
            return x @ R.T, R, None
        else:
            return x, None, None
    
    def transform_vectors(self, vectors, R=None, scale=None):
        """Transform vector field values (for equivariance)"""
        if R is not None and vectors.shape[-1] == 2:
            return vectors @ R.T
        return vectors
    
    def __call__(self, x):
        """For backward compatibility"""
        transformed_x, _, _ = self.transform_points(x)
        return transformed_x


class ScalingTransformation:
    """Pure scaling transformation"""
    def __init__(self, scale_range=(0.8, 1.2)):
        self.scale_min, self.scale_max = scale_range
    
    def transform_points(self, x):
        """Transform input points"""
        scale = torch.rand(x.shape[0], 1, device=x.device, dtype=x.dtype) * (self.scale_max - self.scale_min) + self.scale_min
        return x * scale, None, scale
    
    def transform_vectors(self, vectors, R=None, scale=None):
        """Transform vector field values (for equivariance)"""
        if scale is not None:
            return vectors * scale
        return vectors
    
    def __call__(self, x):
        """For backward compatibility"""
        transformed_x, _, _ = self.transform_points(x)
        return transformed_x
