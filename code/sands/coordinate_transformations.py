""" Some functions for moving coordiantes around. """

import torch
import numpy as np
from typing import Optional, Callable, Tuple


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


def stereographic_projection_to_sphere(x_f: torch.Tensor, r: float):
    """Lift points from flat space to sphere using stereographic projection."""
    ## TODO clean up this func, pprefer to not pass in c at all. always use c=0 for this func
    n = x_f.shape[1]
    
    d_squared = torch.sum(x_f**2, dim=1, keepdim=True)
    scale = (2 * r**2) / (d_squared + r**2)
    
    x_s_first_n = scale * x_f
    height = r * (d_squared - r**2) / (r**2 + d_squared)
    x_s_last = height
    
    x_s = torch.cat([x_s_first_n, x_s_last], dim=1)
    return x_s


def central_projection_to_flat(x_h: torch.Tensor, c: torch.Tensor):
    """Map points from higher-dimensional space to flat space via central projection."""
    directions = x_h - c
    scaling = -c[-1] / (directions[:, -1] + 1e-10)
    x_f_homo = c + scaling.unsqueeze(1) * directions
    x_f = x_f_homo[:, :-1]
    return x_f


def stereographic_projection_to_flat(x_s: torch.Tensor, r: float):
    """Project points from sphere back to flat space."""
    north_pole = torch.zeros(x_s.shape[1], device=x_s.device)
    north_pole[-1] = r
    return central_projection_to_flat(x_s, north_pole)


def warp_with_rotation(points: torch.Tensor, r: float, rotation_matrix: torch.Tensor):
    """Apply stereographic-rotation-stereographic sequence to n-dimensional points."""
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32)
    if not isinstance(rotation_matrix, torch.Tensor):
        rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32)
    
    n = points.shape[1]
    
    # Step 1: Map to n-sphere using stereographic projection
    sphere_points = stereographic_projection_to_sphere(points, 1.0)
    
    # Step 2: Apply rotation
    rotated_points = torch.matmul(sphere_points, rotation_matrix.transpose(0, 1))
    
    # Step 3: Scale and project back to flat space
    rotated_points = rotated_points * r  # Fixed typo from original
    flat_points = stereographic_projection_to_flat(rotated_points, r)
    
    return flat_points


def create_stereographic_transform(radius: float = 1.0, 
                                  rotation_angles: Optional[list] = None,
                                  n_dims: int = 2,
                                  device: str = 'cpu') -> Tuple[Callable, Callable]:
    """
    Create pullback/pushforward functions using stereographic projection with rotation.
    
    Args:
        radius: Radius parameter for stereographic projection
        rotation_angles: List of rotation angles for n-dimensional rotation
        n_dims: Number of dimensions
        device: PyTorch device
        
    Returns:
        (pullback_fn, pushforward_fn) tuple
    """
    if rotation_angles is None:
        # Default to no rotation (identity)
        num_angles = n_dims * (n_dims + 1) // 2  # Number of rotation planes in (n+1)D
        rotation_angles = [0.0] * num_angles
    
    # Create rotation matrix for (n+1)-dimensional space
    rotation_matrix = create_nd_rotation_matrix(rotation_angles, n_dims + 1)
    rotation_matrix = rotation_matrix.to(device)
    
    # Create inverse rotation
    rotation_matrix_inv = rotation_matrix.transpose(0, 1)
    
    def pullback_fn(x: torch.Tensor) -> torch.Tensor:
        """Global to local using inverse stereographic warp."""
        return warp_with_rotation(x, radius, rotation_matrix_inv)
    
    def pushforward_fn(x: torch.Tensor) -> torch.Tensor:
        """Local to global using stereographic warp."""
        return warp_with_rotation(x, radius, rotation_matrix)
    
    return pullback_fn, pushforward_fn


def create_identity_transform() -> Tuple[Callable, Callable]:
    """Create identity transformation functions."""
    identity = lambda x: x
    return identity, identity
