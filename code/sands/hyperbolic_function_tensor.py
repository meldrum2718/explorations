import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from typing import Optional
import math

class HyperbolicFunctionTensor:
    """
    Hyperbolic lens function with affine positioning using PyTorch.
    Stores function values on a regular meshgrid over [-1,1]^n (ball coordinates).
    """
    
    def __init__(self, resolution: int, n_dims: int = 2, n_channels: int = 1, 
                 affine_transform: Optional[torch.Tensor] = None, 
                 center: Optional[torch.Tensor] = None, device: str = 'cpu'):
        """
        Args:
            resolution: Grid resolution per dimension
            n_dims: Spatial dimensions
            n_channels: Number of function channels
            affine_transform: (n_dims, n_dims) matrix A for coordinate frame
            center: Translation offset for the lens
            device: torch device
        """
        self.resolution = resolution
        self.n_dims = n_dims
        self.n_channels = n_channels
        self.device = device
        
        # Affine transformation
        if affine_transform is None:
            self.A = torch.eye(n_dims, device=device, dtype=torch.float32)
        else:
            self.A = affine_transform.to(device)
        self.A_inv = torch.linalg.pinv(self.A)
        
        # Center offset
        if center is None:
            self.center = torch.zeros(n_dims, device=device, dtype=torch.float32)
        else:
            self.center = center.to(device)
        
        # Create meshgrid over [-1, 1]^n
        self.mesh_ball = self._create_meshgrid()
        self.values = torch.zeros(*self.mesh_ball.shape[:-1], n_channels, device=device)
    
    def _create_meshgrid(self) -> torch.Tensor:
        """Create regular meshgrid over [-1, 1]^n for arbitrary dimensions."""
        coords = torch.linspace(-1, 1, self.resolution, device=self.device, dtype=torch.float32)
        
        # Create n-dimensional meshgrid
        coord_arrays = torch.meshgrid(*[coords] * self.n_dims, indexing='ij')
        
        # Stack coordinate arrays along the last dimension
        return torch.stack(coord_arrays, dim=-1)
    
    def b2f(self, x: torch.Tensor) -> torch.Tensor:
        """Ball to flat: A(atanh(||x||) * (x/||x||)) + center"""
        # Only process points inside unit ball
        r = torch.linalg.norm(x, dim=-1, keepdim=True)
        r = torch.clamp(r, min=1e-8, max=1-1e-8)
        
        unit_x = x / r
        flat_coords = torch.arctanh(r) * unit_x
        
        # Apply affine transform and add center offset
        transformed = (self.A @ flat_coords.unsqueeze(-1)).squeeze(-1)
        return transformed + self.center
    
    def f2b(self, x: torch.Tensor) -> torch.Tensor:
        """Flat to ball: tanh(||A_inv(x - center)||) * (A_inv(x - center)/||A_inv(x - center)||)"""
        # Subtract center first, then apply inverse transform
        centered_coords = x - self.center
        local_coords = (self.A_inv @ centered_coords.unsqueeze(-1)).squeeze(-1)
        r = torch.linalg.norm(local_coords, dim=-1, keepdim=True)
        r = torch.clamp(r, min=1e-8)
        
        unit_coords = local_coords / r
        ball_r = torch.tanh(r)
        
        return ball_r * unit_coords
    
    def sample_at(self, points: torch.Tensor) -> torch.Tensor:
        """Sample function at global coordinates by transforming to local frame."""
        local_ball_coords = self.f2b(points)
        return self._interpolate_nd(local_ball_coords)
    
    def _interpolate_nd(self, query_points: torch.Tensor) -> torch.Tensor:
        """N-dimensional linear interpolation on regular grid."""
        batch_size = query_points.shape[0]
        
        # Transform query points from [-1, 1] to [0, resolution-1] grid coordinates
        grid_coords = (query_points + 1) * (self.resolution - 1) / 2
        
        # Clamp to valid range
        grid_coords = torch.clamp(grid_coords, 0, self.resolution - 1)
        
        # Get integer and fractional parts
        grid_coords_floor = torch.floor(grid_coords).long()
        grid_coords_frac = grid_coords - grid_coords_floor.float()
        
        # Clamp floor coordinates to valid indices
        grid_coords_floor = torch.clamp(grid_coords_floor, 0, self.resolution - 2)
        
        # Generate all 2^n_dims corner points for each query point
        n_corners = 2 ** self.n_dims
        corner_offsets = torch.zeros(n_corners, self.n_dims, dtype=torch.long, device=self.device)
        
        # Generate binary combinations for corner offsets
        for i in range(n_corners):
            for d in range(self.n_dims):
                corner_offsets[i, d] = (i >> d) & 1
        
        # Compute interpolation weights and sample values
        result = torch.zeros(batch_size, self.n_channels, device=self.device)
        
        for corner_idx in range(n_corners):
            # Get corner coordinates
            corner_coords = grid_coords_floor + corner_offsets[corner_idx]
            
            # Compute interpolation weight for this corner
            weight = torch.ones(batch_size, device=self.device)
            for d in range(self.n_dims):
                if corner_offsets[corner_idx, d] == 0:
                    weight *= (1 - grid_coords_frac[:, d])
                else:
                    weight *= grid_coords_frac[:, d]
            
            # Sample values at corner coordinates
            corner_values = self._sample_at_grid_indices(corner_coords)
            
            # Accumulate weighted contribution
            result += weight.unsqueeze(1) * corner_values
        
        return result
    
    def _sample_at_grid_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Sample values at integer grid indices."""
        batch_size = indices.shape[0]
        
        # Convert n-dimensional indices to flat indices
        if self.n_dims == 1:
            flat_indices = indices[:, 0]
        elif self.n_dims == 2:
            flat_indices = indices[:, 0] * self.resolution + indices[:, 1]
        elif self.n_dims == 3:
            flat_indices = (indices[:, 0] * self.resolution + indices[:, 1]) * self.resolution + indices[:, 2]
        else:
            # General case for higher dimensions
            flat_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            multiplier = 1
            for d in range(self.n_dims - 1, -1, -1):
                flat_indices += indices[:, d] * multiplier
                multiplier *= self.resolution
        
        # Reshape values to flat array and index
        values_flat = self.values.reshape(-1, self.n_channels)
        return values_flat[flat_indices]
    
    @classmethod
    def from_function(cls, func, resolution: int, n_dims: int = 2, n_channels: int = 1,
                     affine_transform: Optional[torch.Tensor] = None,
                     center: Optional[torch.Tensor] = None,
                     device: str = 'cpu') -> 'HyperbolicFunctionTensor':
        """Create by sampling a function R^n -> R^c."""
        hf = cls(resolution, n_dims, n_channels, affine_transform, center, device)
        
        # Transform mesh to flat coordinates and sample function
        mesh_flat = hf.mesh_ball.reshape(-1, n_dims)
        sample_points = hf.b2f(mesh_flat)
        
        # Evaluate function
        sample_values = func(sample_points)
        if isinstance(sample_values, np.ndarray):
            sample_values = torch.tensor(sample_values, device=device, dtype=torch.float32)
        if sample_values.dim() == 1:
            sample_values = sample_values.unsqueeze(1)
        
        hf.values = sample_values.reshape(*hf.mesh_ball.shape[:-1], n_channels)
        return hf
    
    @classmethod
    def positioned_at(cls, center: torch.Tensor, scale: float = 1.0, 
                     rotation_angle: float = 0.0, resolution: int = 32,
                     n_dims: int = 2, n_channels: int = 1, device: str = 'cpu') -> 'HyperbolicFunctionTensor':
        """Factory for common positioning: translation + scale + rotation."""
        if isinstance(center, (list, np.ndarray)):
            center = torch.tensor(center, device=device, dtype=torch.float32)
        
        n_dims = len(center)
        
        # Create rotation matrix
        if n_dims == 2:
            cos_a, sin_a = math.cos(rotation_angle), math.sin(rotation_angle)
            rotation = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], device=device, dtype=torch.float32)
        else:
            rotation = torch.eye(n_dims, device=device, dtype=torch.float32)
        
        # Affine transform: scale * rotation
        A = scale * rotation
        
        return cls(resolution=resolution, n_dims=n_dims, n_channels=n_channels,
                  affine_transform=A, center=center, device=device)
    
    def render_2d(self, resolution: int = 256, extent: float = 3.0) -> torch.Tensor:
        """Render 2D function as image by sampling on regular grid."""
        if self.n_dims != 2:
            raise ValueError("render_2d only works for 2D functions")
        
        # Create regular grid
        x = torch.linspace(-extent, extent, resolution, device=self.device, dtype=torch.float32)
        y = torch.linspace(-extent, extent, resolution, device=self.device, dtype=torch.float32)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1)
        
        # Sample function at grid points
        values = self.sample_at(grid_points)
        
        # Reshape to image
        if self.n_channels == 1:
            return values[:, 0].reshape(resolution, resolution)
        else:
            return values.reshape(resolution, resolution, self.n_channels)


def demo_function(points: torch.Tensor) -> torch.Tensor:
    """Demo function: sin(f1*x) + sin(f2*y)"""
    f1, f2 = 1.0, 1.0
    x, y = points[:, 0], points[:, 1]
    values = torch.tanh(x**2 + y**2) * (torch.sin(f1 * x) + torch.sin(f2 * y))
    return values.unsqueeze(1)


def main():
    """Interactive demo with sliders for positioning."""
    # Use MPS if available (Apple Silicon), otherwise CUDA, otherwise CPU
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plt.subplots_adjust(bottom=0.30)
    
    # Initial parameters
    center_x, center_y = 0.0, 0.0
    scale = 1.0
    rotation = 0.0
    resolution = 32
    
    def update_plot():
        # Create positioned hyperbolic function
        center = torch.tensor([center_x, center_y], device=device, dtype=torch.float32)
        hf = HyperbolicFunctionTensor.positioned_at(
            center=center, scale=scale, rotation_angle=rotation,
            resolution=int(resolution), n_dims=2, n_channels=1, device=device
        )
        
        # Sample the demo function
        hf = HyperbolicFunctionTensor.from_function(
            demo_function, resolution=int(resolution), n_dims=2, n_channels=1,
            affine_transform=hf.A, center=hf.center, device=device
        )
        
        # Render as images
        image1 = hf.render_2d(resolution=128, extent=3*scale)
        
        # Show the raw hyperbolic function values (ball coordinates)
        image2 = hf.values[:, :, 0]  # First channel
        
        # Update plots
        ax1.clear()
        ax2.clear()
        
        ax1.imshow(image1.cpu().numpy(), extent=[-3, 3, -3, 3], origin='lower', cmap='viridis')
        ax1.set_title('Hyperbolic Function (Positioned)')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        
        ax2.imshow(image2.cpu().numpy(), extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
        ax2.set_title('Raw Values (Ball Coordinates)')
        ax2.set_xlabel('ball x')
        ax2.set_ylabel('ball y')
        
        plt.draw()
    
    # Create sliders
    ax_center_x = plt.axes([0.1, 0.18, 0.3, 0.03])
    ax_center_y = plt.axes([0.1, 0.14, 0.3, 0.03])
    ax_scale = plt.axes([0.1, 0.10, 0.3, 0.03])
    ax_rotation = plt.axes([0.1, 0.06, 0.3, 0.03])
    ax_resolution = plt.axes([0.6, 0.18, 0.3, 0.03])
    
    slider_center_x = Slider(ax_center_x, 'Center X', -50.0, 50.0, valinit=center_x)
    slider_center_y = Slider(ax_center_y, 'Center Y', -50.0, 50.0, valinit=center_y)
    slider_scale = Slider(ax_scale, 'Scale', 0.1, 200.0, valinit=scale)
    slider_rotation = Slider(ax_rotation, 'Rotation', 0, 2*math.pi, valinit=rotation)
    slider_resolution = Slider(ax_resolution, 'Resolution', 8, 1024, valinit=resolution, valfmt='%d')
    
    def update_params(val):
        nonlocal center_x, center_y, scale, rotation, resolution
        center_x = slider_center_x.val
        center_y = slider_center_y.val
        scale = slider_scale.val
        rotation = slider_rotation.val
        resolution = slider_resolution.val
        update_plot()
    
    slider_center_x.on_changed(update_params)
    slider_center_y.on_changed(update_params)
    slider_scale.on_changed(update_params)
    slider_rotation.on_changed(update_params)
    slider_resolution.on_changed(update_params)
    
    # Initial plot
    update_plot()
    plt.show()


if __name__ == "__main__":
    main()
