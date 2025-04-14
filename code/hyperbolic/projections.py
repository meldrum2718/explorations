"""
Some functions for working with hyperbolic space.

Supporting various transformations between flat space, spheres, and balls.
It also includes utilities for rotating points in higher-dimensional spaces.
"""

import torch
import numpy as np
from scipy.interpolate import LinearNDInterpolator

def central_projection_to_hemisphere(x_f: torch.Tensor, c: torch.Tensor, r: float):
    """
    Lifts each point x_f in R^{n} to the point x_h in R^{n+1} on the hemisphere
    of radius r centered at c = [0, ..., 0, r] such that x_f, x_h, and c are collinear.
    
    Parameters:
    -----------
    x_f : torch.Tensor
        Points in flat space R^n, shape (batch_size, n)
    c : torch.Tensor
        Center of the hemisphere, shape (n+1,)
    r : float
        Radius of the hemisphere
        
    Returns:
    --------
    x_h : torch.Tensor
        Points on the hemisphere, shape (batch_size, n+1)
    """
    batch_size = x_f.shape[0]
    n = x_f.shape[1]
    
    # Create homogeneous coordinates by appending zeros
    x_f_homo = torch.cat([x_f, torch.zeros(batch_size, 1, device=x_f.device)], dim=1)
    
    # Compute vectors from center to each point
    x_centered = x_f_homo - c
    
    # Direction vectors from c to each x_f (extended to R^{n+1})
    directions = x_centered / (torch.norm(x_centered, dim=1, keepdim=True) + 1e-10)
    
    # Generate hemisphere points by moving r units along the directions
    x_h = c + r * directions
    
    # Ensure all points are on the correct hemisphere (last coordinate > c[n])
    # For a hemisphere centered at c, we want the north hemisphere where the last coordinate > c[n-1]
    last_coord_offset = x_h[:, -1] - c[-1]
    mask = last_coord_offset > 0
    if mask.any():
        # If any points are on the wrong hemisphere, flip them
        x_h[mask] = 2*c - x_h[mask]
    
    return x_h

def central_projection_to_flat(x_h: torch.Tensor, c: torch.Tensor):
    """
    Maps each point x_h in R^{n+1} to the point x_f in R^{n} × {0} such that
    x_f, x_h, and c are collinear.
    
    Parameters:
    -----------
    x_h : torch.Tensor
        Points on the hemisphere, shape (batch_size, n+1)
    c : torch.Tensor
        Center point for the projection, shape (n+1,)
        
    Returns:
    --------
    x_f : torch.Tensor
        Points in flat space, shape (batch_size, n)
    """
    # Direction vectors from c to x_h
    directions = x_h - c
    
    # Calculate the scaling factor for each direction
    # We want to find t such that c + t*directions has last coordinate = 0
    # This means c[-1] + t*directions[:, -1] = 0
    # So t = -c[-1] / directions[:, -1]
    scaling = -c[-1] / (directions[:, -1] + 1e-10)
    
    # Apply scaling to find the intersection with the flat space
    x_f_homo = c + scaling.unsqueeze(1) * directions
    
    # Extract the first n coordinates (discard the last coordinate which should be 0)
    x_f = x_f_homo[:, :-1]
    
    return x_f

def stereographic_projection_to_sphere(x_f: torch.Tensor, c: torch.Tensor, r: float):
    """
    Lifts points from flat space to sphere using stereographic projection.
    
    Parameters:
    -----------
    x_f : torch.Tensor
        Points in flat space R^n, shape (batch_size, n)
    c : torch.Tensor
        Center point for the projection, shape (n+1,)
    r : float
        Radius of the sphere
        
    Returns:
    --------
    x_s : torch.Tensor
        Points on the sphere, shape (batch_size, n+1)
    """
    batch_size = x_f.shape[0]
    n = x_f.shape[1]
    
    # First n coordinates of the center
    c_n = c[:-1]
    
    # Calculate squared distance from each point to the center projection
    d_squared = torch.sum((x_f - c_n)**2, dim=1, keepdim=True)
    
    # Calculate scaling factor for stereographic projection
    # For standard stereographic projection:
    # The plane point (x - c_n) maps to sphere point with first n coordinates
    # proportional to (x - c_n) and scaled by factor 2r²/(d² + r²)
    scale = (2 * r**2) / (d_squared + r**2)
    
    # Calculate first n coordinates on sphere
    x_s_first_n = c_n + scale * (x_f - c_n)
    
    # Calculate height (last coordinate)
    # For points at distance d from center on plane, height from equator is:
    # h = r * (d² - r²)/(r² + d²)
    height = r * (d_squared - r**2) / (r**2 + d_squared)
    x_s_last = c[-1] + height
    
    # Combine coordinates
    x_s = torch.cat([x_s_first_n, x_s_last], dim=1)
    
    return x_s

def stereographic_projection_to_flat(x_s: torch.Tensor, c: torch.Tensor, r: float):
    """
    Projects points from sphere back to flat space using inverse stereographic projection.
    
    Parameters:
    -----------
    x_s : torch.Tensor
        Points on the sphere, shape (batch_size, n+1)
    c : torch.Tensor
        Center of the sphere, shape (n+1,)
    r : float
        Radius of the sphere
        
    Returns:
    --------
    x_f : torch.Tensor
        Points in flat space, shape (batch_size, n)
    """
    batch_size = x_s.shape[0]
    n = x_s.shape[1] - 1
    
    # Calculate North Pole position
    north_pole = c.clone()
    north_pole[-1] = c[-1] + r

    return central_projection_to_flat(x_s, north_pole)

def ball_to_flat(x_p: torch.Tensor, eccentricity: float = 1.0, c_factor: float = 1.34):
    """
    Maps a point x_p from the unit ball in R^n to R^n via:
    1. Stereographic projection to the unit sphere centered at [0,...,0,1]
    2. Central projection back to flat space
    
    Parameters:
    -----------
    x_p : torch.Tensor
        Points in the unit ball, shape (batch_size, n)
    eccentricity : float
        Scaling factor for all coordinates except the last one (z-coordinate)
    c_factor : float
        Factor to scale the center point for the central projection
        
    Returns:
    --------
    x_f : torch.Tensor
        Points in flat space, shape (batch_size, n)
    """
    # Set projection parameters
    n = x_p.shape[1]
    c = torch.zeros(n+1, device=x_p.device)
    c[-1] = 1.0  # Center at [0,...,0,1]
    r = 1.0      # Unit radius
    
    # Step 1: Stereographic projection to sphere
    sphere_points = stereographic_projection_to_sphere(x_p, c, r)
    
    # Apply eccentricity scaling to all but the last coordinate
    sphere_points[:, :-1] *= eccentricity
    
    # Step 2: Central projection to flat space with scaled center
    flat_points = central_projection_to_flat(sphere_points, c_factor * c)
    
    return flat_points

def flat_to_ball(x_f: torch.Tensor, eccentricity: float = 3.0, c_factor: float = 1.34):
    """
    Maps a point x_f from R^n to the unit ball in R^n via:
    1. Central projection to the unit hemisphere centered at [0,...,0,1]
    2. Stereographic projection back to flat space (which maps to the ball)
    
    Parameters:
    -----------
    x_f : torch.Tensor
        Points in flat space, shape (batch_size, n)
    eccentricity : float
        Scaling factor that should match the one used in ball_to_flat
    c_factor : float
        Factor to scale the center point (should match ball_to_flat)
        
    Returns:
    --------
    x_p : torch.Tensor
        Points in the unit ball, shape (batch_size, n)
    """
    # Set projection parameters
    n = x_f.shape[1]
    c = torch.zeros(n+1, device=x_f.device)
    c[-1] = 1.0  # Center at [0,...,0,1]
    r = 1.0      # Unit radius
    
    # Step 1: Central projection to hemisphere
    hemisphere_points = central_projection_to_hemisphere(x_f, c_factor * c, r)
    
    # Apply inverse eccentricity scaling to all but the last coordinate
    if eccentricity != 1.0:
        hemisphere_points[:, :-1] /= eccentricity
    
    # Step 2: Stereographic projection to flat space (maps to the unit ball)
    ball_points = stereographic_projection_to_flat(hemisphere_points, c, r)
    
    return ball_points

def create_nd_rotation_matrix(angles, dim):
    """
    Create a generalized rotation matrix for n-dimensional space.
    
    In n-dimensional space, rotations occur in 2D planes. There are n(n-1)/2 
    possible planes of rotation in an n-dimensional space. For example:
    - In 2D: 1 plane (xy)
    - In 3D: Possible rotation planes: (xy, xz, yz)
    - In 4D: Possible rotation planes: (xy, xz, yz, xw, yw, zw)
    
    Parameters:
    -----------
    angles : list or tensor
        List of rotation angles (in radians) for each rotation plane.
        For an n-dimensional space, there are n(n-1)/2 possible rotation planes.
        The expected order is: [θ_01, θ_02, ..., θ_0(n-1), θ_12, θ_13, ..., θ_(n-2)(n-1)]
        where θ_ij represents rotation in the plane defined by axes i and j.
    dim : int
        The dimensionality of the space.
    
    Returns:
    --------
    torch.Tensor
        (dim)x(dim) rotation matrix
    """
    # Start with identity matrix
    rot_matrix = torch.eye(dim)
    
    # Convert angles to tensor if not already
    if not isinstance(angles, torch.Tensor):
        angles = torch.tensor(angles, dtype=torch.float32)
    
    # Apply rotations for each plane
    angle_idx = 0
    for i in range(dim-1):
        for j in range(i+1, dim):
            if angle_idx < len(angles):
                # Create a rotation in the i-j plane
                theta = angles[angle_idx]
                plane_rot = torch.eye(dim)
                plane_rot[i, i] = torch.cos(theta)
                plane_rot[i, j] = -torch.sin(theta)
                plane_rot[j, i] = torch.sin(theta)
                plane_rot[j, j] = torch.cos(theta)
                
                # Apply this rotation
                rot_matrix = torch.matmul(plane_rot, rot_matrix)
                angle_idx += 1
    
    return rot_matrix

def warp_with_rotation(points: torch.Tensor, r: float, rotation_matrix: torch.Tensor):
    """
    Apply a stereographic-rotation-stereographic sequence to n-dimensional points.
    
    Parameters:
    -----------
    points : torch.Tensor
        Points in R^n, shape (n_points, n)
    r : float
        Radius parameter for the stereographic projection
    rotation_matrix : torch.Tensor
        (n+1)x(n+1) rotation matrix for rotating points on the n-sphere
        
    Returns:
    --------
    torch.Tensor
        Transformed points in R^n
    """
    # Ensure inputs are torch tensors
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32)
    if not isinstance(rotation_matrix, torch.Tensor):
        rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32)
    
    n = points.shape[1]  # Dimensionality of input
    
    # Step 1: Map each point to a point on the n-sphere using stereographic projection
    # Set up center point for projection
    c = torch.zeros(n + 1, device=points.device) # center at origin
    
    # Project points to sphere
    sphere_points = stereographic_projection_to_sphere(points, c, 1)
    
    # Step 2: Apply rotation
    rotated_points = torch.matmul(sphere_points, rotation_matrix.transpose(0, 1))

    # scale points by radius
    ## TODO
    
    # Step 3: Stereographic projection back to flat space
    flat_points = stereographic_projection_to_flat(rotated_points, c, r)
    
    return flat_points

def apply_warp_to_coords(coords, r, rotation_angles, eccentricity, c_factor):
    """
    Apply the warping sequence to input coordinates.
    """
    device = coords.device
    
    # Create rotation matrix on the same device as coords
    rotation_matrix = create_nd_rotation_matrix(rotation_angles, 3).to(device)
    
    flat_points = ball_to_flat(coords, eccentricity, c_factor)
    warped_points = warp_with_rotation(flat_points, r, rotation_matrix)
    return warped_points

def normalize_tensor(tensor):
    """
    Normalize a tensor to have values between 0 and 1.
    
    Parameters:
    -----------
    tensor : torch.Tensor
        Tensor to normalize
        
    Returns:
    --------
    normalized_tensor : torch.Tensor
        Normalized tensor with values between 0 and 1
    """
    with torch.no_grad():
        min_val = tensor.min()
        max_val = tensor.max()
        
        # Avoid division by zero
        if max_val - min_val < 1e-8:
            return torch.zeros_like(tensor)
        
        normalized = (tensor - min_val) / (max_val - min_val)
        
    return normalized


class FunctionTensor:
    """
    A class to represent a function tensor defined on a grid with coordinate transformations
    between ball space and flat space.
    """
    
    def __init__(self, resolution=200, domain_size=1.0, eccentricity=3.0, c_factor=1.34, device='cpu'):
        """
        Initialize the function tensor with a grid of points.
        
        Parameters:
        -----------
        resolution : int
            Grid resolution
        domain_size : float
            Size of the domain (half-width and half-height), defaults to 1.0 for unit ball
        eccentricity : float
            Eccentricity parameter for ball_to_flat transformation
        c_factor : float
            C-factor parameter for ball_to_flat transformation
        device : torch.device or str
            Device to put the tensor on
        """
        self.resolution = resolution
        self.domain_size = domain_size
        self.eccentricity = eccentricity
        self.c_factor = c_factor
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        # Create grid and function values
        self._create_grid()
        self._compute_function_values()
        
        # Initialize interpolators
        self.ball_interpolator = None
        self.flat_interpolator = None
        
    def _create_grid(self):
        """Create grid coordinates in ball space and flat space."""
        with torch.no_grad():
            # Create grid coordinates in ball space
            x = torch.linspace(-self.domain_size, self.domain_size, self.resolution)
            y = torch.linspace(-self.domain_size, self.domain_size, self.resolution)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            
            # Create coordinates tensor for ball space
            self.ball_coords = torch.stack([X.flatten(), Y.flatten()], dim=-1).to(self.device)
            
            # Calculate the norm of each coordinate for masking
            self.norms = torch.sqrt(self.ball_coords[:, 0]**2 + self.ball_coords[:, 1]**2)
            
            # Valid points are inside the circle
            self.valid_mask = self.norms < self.domain_size
            
            # Convert ball space coordinates to flat space
            self.flat_coords = ball_to_flat(self.ball_coords, self.eccentricity, self.c_factor)
    
    def _compute_function_values(self):
        """Compute function values using the flat space coordinates."""
        with torch.no_grad():
            # Example function: sin(xy) + cos(2y) + sin(3x)
            # This can be replaced with any other function
            values = torch.sin(self.flat_coords[:, 0] * self.flat_coords[:, 1]) + \
                    torch.cos(2 * self.flat_coords[:, 1]) + \
                    torch.sin(3 * self.flat_coords[:, 0])
            
            # Zero out values outside the valid domain
            self.values = values * self.valid_mask
            
            # Reshape to 2D tensor for grid representation
            self.function_tensor = self.values.reshape(self.resolution, self.resolution)
            
            # Always normalize the tensor when creating it
            self.function_tensor = normalize_tensor(self.function_tensor)
            self.values = self.function_tensor.flatten()
    
    def set_function_values(self, func):
        """
        Set function values using a custom function that takes flat coordinates as input.
        
        Parameters:
        -----------
        func : callable
            Function that takes flat coordinates tensor as input and returns function values
        """
        with torch.no_grad():
            # Apply function to flat coordinates
            values = func(self.flat_coords)
            
            # Zero out values outside the valid domain
            self.values = values * self.valid_mask
            
            # Reshape to 2D tensor
            self.function_tensor = self.values.reshape(self.resolution, self.resolution)
            
            # Always normalize
            self.function_tensor = normalize_tensor(self.function_tensor)
            self.values = self.function_tensor.flatten()
        
        # Reset interpolators
        self.ball_interpolator = None
        self.flat_interpolator = None
    
    def set_values_from_tensor(self, tensor):
        """
        Set function values directly from a tensor.
        
        Parameters:
        -----------
        tensor : torch.Tensor
            2D tensor with shape (resolution, resolution) containing function values
        """
        if tensor.shape != (self.resolution, self.resolution):
            raise ValueError(f"Expected tensor of shape ({self.resolution}, {self.resolution}), "
                            f"got {tensor.shape}")
        
        with torch.no_grad():
            self.function_tensor = tensor.to(self.device)
            self.values = self.function_tensor.flatten()
            
            # Apply mask
            self.values = self.values * self.valid_mask
            
            # Update function tensor with masked values
            self.function_tensor = self.values.reshape(self.resolution, self.resolution)
        
        # Reset interpolators
        self.ball_interpolator = None
        self.flat_interpolator = None
    
    def _ensure_ball_interpolator(self):
        """Create interpolator for ball space if it doesn't exist."""
        if self.ball_interpolator is None:
            # Convert tensors to numpy for interpolation
            ball_coords_np = self.ball_coords.detach().cpu().numpy()
            values_np = self.values.detach().cpu().numpy()
            
            # Create interpolator
            self.ball_interpolator = LinearNDInterpolator(
                ball_coords_np, 
                values_np, 
                fill_value=0
            )
    
    def _ensure_flat_interpolator(self):
        """Create interpolator for flat space if it doesn't exist."""
        if self.flat_interpolator is None:
            # Convert tensors to numpy for interpolation
            flat_coords_np = self.flat_coords.detach().cpu().numpy()
            values_np = self.values.detach().cpu().numpy()
            
            # Create interpolator
            self.flat_interpolator = LinearNDInterpolator(
                flat_coords_np, 
                values_np, 
                fill_value=0
            )
    
    def sample_at_ball_coords(self, coords):
        """
        Sample the function at given ball space coordinates using interpolation.
        
        Parameters:
        -----------
        coords : torch.Tensor
            Coordinates in ball space to sample at, shape (N, 2)
            
        Returns:
        --------
        sampled_values : torch.Tensor
            Function values at the given coordinates
        """
        self._ensure_ball_interpolator()
        
        # Convert coords to numpy
        coords_np = coords.detach().cpu().numpy()
        
        # Sample function
        sampled_values = self.ball_interpolator(coords_np)
        
        # Convert back to tensor
        return torch.tensor(sampled_values, dtype=torch.float32, device=self.device)
    
    def sample_at_flat_coords(self, coords):
        """
        Sample the function at given flat space coordinates using interpolation.
        
        Parameters:
        -----------
        coords : torch.Tensor
            Coordinates in flat space to sample at, shape (N, 2)
            
        Returns:
        --------
        sampled_values : torch.Tensor
            Function values at the given coordinates
        """
        self._ensure_flat_interpolator()
        
        # Convert coords to numpy
        coords_np = coords.detach().cpu().numpy()
        
        # Sample function
        sampled_values = self.flat_interpolator(coords_np)
        
        # Convert back to tensor
        return torch.tensor(sampled_values, dtype=torch.float32, device=self.device)
    
    def get_normalized_tensor(self):
        """
        Get a normalized version of the function tensor with values between 0 and 1.
        
        Returns:
        --------
        normalized_tensor : torch.Tensor
            Normalized function tensor
        """
        return normalize_tensor(self.function_tensor)
    
    def warp_coordinates(self, coords, r=1.0, rotation_angles=None):
        """
        Apply warping to the coordinates.
        
        Parameters:
        -----------
        coords : torch.Tensor
            Coordinates to warp, shape (N, 2)
        r : float
            Radius parameter for the stereographic projection
        rotation_angles : list or torch.Tensor
            Rotation angles for 3D rotation matrix
            
        Returns:
        --------
        warped_coords : torch.Tensor
            Warped coordinates
        """
        if rotation_angles is None:
            rotation_angles = [0.0, 0.0, 0.0]  # Default: no rotation
            
        return apply_warp_to_coords(
            coords, 
            r, 
            rotation_angles, 
            self.eccentricity, 
            self.c_factor
        )
    
    def warp_and_sample(self, coords, r=1.0, rotation_angles=None, use_flat_space=False):
        """
        Warp coordinates and sample function values.
        
        Parameters:
        -----------
        coords : torch.Tensor
            Coordinates to warp and sample at, shape (N, 2)
        r : float
            Radius parameter for the stereographic projection
        rotation_angles : list or torch.Tensor
            Rotation angles for 3D rotation matrix
        use_flat_space : bool
            If True, treats input coordinates as flat space coords,
            otherwise treats them as ball space coords
            
        Returns:
        --------
        sampled_values : torch.Tensor
            Function values at the warped coordinates
        """
        warped_coords = self.warp_coordinates(coords, r, rotation_angles)
        
        if use_flat_space:
            return self.sample_at_flat_coords(warped_coords)
        else:
            # Convert warped flat coords to ball coords
            ball_coords = flat_to_ball(
                warped_coords, 
                self.eccentricity, 
                self.c_factor
            )
            return self.sample_at_ball_coords(ball_coords)
    
    def visualize(self, ax=None, cmap='viridis', normalize=True):
        """
        Visualize the function tensor.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        cmap : str, optional
            Colormap name
        normalize : bool, optional
            Whether to normalize the tensor values
            
        Returns:
        --------
        ax : matplotlib.axes.Axes
            Axes with the plot
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))
        
        # Get tensor for visualization
        tensor = self.get_normalized_tensor() if normalize else self.function_tensor
        
        # Plot
        im = ax.imshow(
            tensor.detach().cpu().numpy(),
            extent=[-self.domain_size, self.domain_size, -self.domain_size, self.domain_size],
            origin='lower',
            cmap=cmap
        )
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Set labels
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Function Tensor')
        
        return ax
