"""
TODO make this code more general (i.e. arbitrary nd stereographic proj)
move technical methods to projections.py, have this file just for vis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

# Import the projection functions
from projections import (
    ball_to_flat,
    flat_to_ball
)

def create_rotation_matrix(theta_x, theta_y, theta_z):
    """
    Create a 3D rotation matrix combining rotations around x, y, and z axes.
    
    Parameters:
    -----------
    theta_x, theta_y, theta_z : float
        Rotation angles in radians around x, y, and z axes respectively
    
    Returns:
    --------
    numpy.ndarray
        3x3 rotation matrix
    """
    # Rotation around x-axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])
    
    # Rotation around y-axis
    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])
    
    # Rotation around z-axis
    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation: first around x, then y, then z
    return np.dot(Rz, np.dot(Ry, Rx))

def warp_with_rotation(points, r, Rot):
    """
    Apply a stereographic-rotation-stereographic sequence to points.
    
    Parameters:
    -----------
    points : numpy.ndarray
        Points in R^2, shape (n_points, 2)
    r : float
        Radius parameter for the stereographic projection
    Rot : numpy.ndarray
        3x3 rotation matrix
    
    Returns:
    --------
    numpy.ndarray
        Transformed points in R^2
    """
    # Extract x and y components
    x, y = points[:, 0], points[:, 1]
    
    # Step 1: Map each point (x,y) to a point on the sphere
    denom = x**2 + y**2 + r**2
    X_sphere = 2 * r**2 * x / denom
    Y_sphere = 2 * r**2 * y / denom
    Z_sphere = r * (x**2 + y**2 - r**2) / denom
    
    # Step 2: Apply rotation
    points_3d = np.vstack((X_sphere, Y_sphere, Z_sphere)).T  # Shape: (n_points, 3)
    rotated_points = np.dot(points_3d, Rot.T)  # Apply rotation
    
    # Step 3: Stereographic projection back to plane
    X_rot, Y_rot, Z_rot = rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2]
    
    # Handle points where Z_rot == 1 (division by zero)
    mask = np.abs(1 - Z_rot) > 1e-10
    X_proj = np.zeros_like(X_rot)
    Y_proj = np.zeros_like(Y_rot)
    
    X_proj[mask] = X_rot[mask] / (1 - Z_rot[mask])
    Y_proj[mask] = Y_rot[mask] / (1 - Z_rot[mask])
    
    # Pack the results
    result = np.column_stack((X_proj, Y_proj))
    result[~mask] = 0  # Set points with numerical issues to origin
    
    return result


def apply_sequence_to_function(function, grid_x, grid_y, r, rotation_matrix, eccentricity, c_factor):
    """
    Apply the projection sequence to a function and evaluate it on the grid.
    
    Parameters:
    -----------
    function : callable
        Function that maps R^2 -> R
    grid_x, grid_y : numpy.ndarray
        Grid arrays for x and y dimensions
    r : float
        Radius parameter
    rotation_matrix : numpy.ndarray
        3x3 rotation matrix
    eccentricity : float
        Scaling factor for sphere coordinates
    c_factor : float
        Factor to scale the center point
    
    Returns:
    --------
    numpy.ndarray
        Function values on the transformed grid
    """
    # Create grid of coordinates (resolution x resolution, 2)
    grid_coords = np.stack(np.meshgrid(grid_x, grid_y, indexing='ij'), axis=-1).reshape(-1, 2)
    
    # Apply the sequence
    points_torch = torch.tensor(grid_coords, dtype=torch.float32)
    flat_points = ball_to_flat(points_torch, eccentricity, c_factor).numpy()
    warped_points = warp_with_rotation(flat_points, r, rotation_matrix)
    warped_points_torch = torch.tensor(warped_points, dtype=torch.float32)
    transformed_coords = flat_to_ball(warped_points_torch, eccentricity, c_factor).numpy()

    # Reshape back to grid
    tx = warped_points[:, 0].reshape(len(grid_x), len(grid_y))
    ty = warped_points[:, 1].reshape(len(grid_x), len(grid_y))
    
    # Evaluate function on transformed coordinates
    return function(tx, ty)

def apply_direct_warp(function, grid_x, grid_y, r, rotation_matrix):
    """
    Apply only warp_with_rotation to a function and evaluate it on the grid.
    Skip the ball_to_flat transformation but still do flat_to_ball at the end.
    
    Parameters:
    -----------
    function : callable
        Function that maps R^2 -> R
    grid_x, grid_y : numpy.ndarray
        Grid arrays for x and y dimensions
    r : float
        Radius parameter
    rotation_matrix : numpy.ndarray
        3x3 rotation matrix
    
    Returns:
    --------
    numpy.ndarray
        Function values on the transformed grid
    """
    # Create grid of coordinates (resolution x resolution, 2)
    grid_coords = np.stack(np.meshgrid(grid_x, grid_y, indexing='ij'), axis=-1).reshape(-1, 2)
    
    # Apply only the warp_with_rotation
    warped_points = warp_with_rotation(grid_coords, r, rotation_matrix)
    
    # Reshape back to grid
    tx = warped_points[:, 0].reshape(len(grid_x), len(grid_y))
    ty = warped_points[:, 1].reshape(len(grid_x), len(grid_y))
    
    # Evaluate function on transformed coordinates
    return function(tx, ty)

# Example functions to visualize
def periodic_waves(x, y):
    """Periodic wave pattern over R²"""
    return np.sin(x) * np.sin(y)

def periodic_checkerboard(x, y):
    """Periodic checkerboard pattern over R²"""
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def radial_rings(x, y):
    """Simple radial rings that extend infinitely"""
    r = np.sqrt(x**2 + y**2)
    return np.sin(r)

def complex_periodic(x, y):
    """Complex periodic pattern with multiple frequencies"""
    pattern1 = np.sin(x) * np.sin(y)
    pattern2 = np.sin(2*x) * np.sin(2*y)
    pattern3 = np.sin(0.5*x + 0.5*y)
    
    return (pattern1 + 0.5*pattern2 + 0.3*pattern3) / 1.8

def gaussian_blob(x, y):
    """Gaussian blob centered at origin"""
    return np.exp(-(x**2 + y**2) / 2)

def main():
    # Define functions to visualize
    function_dict = {
        'periodic_checkerboard': periodic_checkerboard,
        'periodic_waves': periodic_waves,
        'radial_rings': radial_rings,
        'complex_periodic': complex_periodic,
        'gaussian_blob': gaussian_blob
    }
    
    # Set up grid parameters
    resolution = 250
    x_max_init = 2  # Initial half-width
    y_max_init = 2  # Initial half-height
    r_init = 2.0  # Initial radius
    eccentricity_init = 3.0  # Initial eccentricity
    c_factor_init = 1.34  # Initial c_factor
    
    # Initial grid (symmetric around zero)
    x = np.linspace(-x_max_init, x_max_init, resolution)
    y = np.linspace(-y_max_init, y_max_init, resolution)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Initial rotation angles (in radians)
    theta_x_init = 0.0
    theta_y_init = 0.0
    theta_z_init = 0.0
    
    # Create initial rotation matrix
    rotation_matrix = create_rotation_matrix(theta_x_init, theta_y_init, theta_z_init)
    
    # Initial function choice
    current_function_name = 'complex_periodic'
    current_function = function_dict[current_function_name]
    
    # Create a figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 7))
    
    # Original function display
    original_im = axs[0].imshow(current_function(X, Y), origin='lower', 
                               extent=[-x_max_init, x_max_init, -y_max_init, y_max_init],
                               cmap='viridis')
    axs[0].set_title('Original Function')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    fig.colorbar(original_im, ax=axs[0])
    
    # Direct warp function display (new plot)
    direct_warp_data = apply_direct_warp(
        current_function, x, y, r_init, rotation_matrix
    )
    direct_warp_im = axs[1].imshow(direct_warp_data, origin='lower', 
                                  extent=[-x_max_init, x_max_init, -y_max_init, y_max_init],
                                  cmap='viridis')
    axs[1].set_title('Direct Warp (Stereographic only)')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    fig.colorbar(direct_warp_im, ax=axs[1])
    
    # Full transformed function display
    transformed_data = apply_sequence_to_function(
        current_function, x, y, r_init, rotation_matrix, eccentricity_init, c_factor_init
    )
    transformed_im = axs[2].imshow(transformed_data, origin='lower', 
                                  extent=[-x_max_init, x_max_init, -y_max_init, y_max_init],
                                  cmap='viridis')
    axs[2].set_title('Full Transform: ball-to-flat → warp → flat-to-ball')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('y')
    fig.colorbar(transformed_im, ax=axs[2])
    
    # Add circles to show boundary
    circle1 = plt.Circle((0, 0), r_init, fill=False, color='red', linestyle='--')
    circle2 = plt.Circle((0, 0), r_init, fill=False, color='red', linestyle='--')
    circle3 = plt.Circle((0, 0), r_init, fill=False, color='red', linestyle='--')
    axs[0].add_patch(circle1)
    axs[1].add_patch(circle2)
    axs[2].add_patch(circle3)
    
    # Adjust the layout to make room for sliders
    plt.subplots_adjust(bottom=0.55)  # Increased bottom margin to accommodate sliders
    
    # Add sliders for parameters
    ax_radius = plt.axes([0.25, 0.45, 0.65, 0.03])
    ax_theta_x = plt.axes([0.25, 0.40, 0.65, 0.03])
    ax_theta_y = plt.axes([0.25, 0.35, 0.65, 0.03])
    ax_theta_z = plt.axes([0.25, 0.30, 0.65, 0.03])
    ax_x_max = plt.axes([0.25, 0.25, 0.65, 0.03])  # Width control (x_max)
    ax_y_max = plt.axes([0.25, 0.20, 0.65, 0.03])  # Height control (y_max)
    ax_eccentricity = plt.axes([0.25, 0.15, 0.65, 0.03])  # Moved up
    ax_c_factor = plt.axes([0.25, 0.10, 0.65, 0.03])  # Moved up
    
    slider_radius = Slider(ax_radius, 'Radius', 0.005, 100.0, valinit=r_init)
    slider_theta_x = Slider(ax_theta_x, 'X Rotation', -np.pi, np.pi, valinit=theta_x_init, valstep=0.05)
    slider_theta_y = Slider(ax_theta_y, 'Y Rotation', -np.pi, np.pi, valinit=theta_y_init, valstep=0.05)
    slider_theta_z = Slider(ax_theta_z, 'Z Rotation', -np.pi, np.pi, valinit=theta_z_init, valstep=0.05)
    slider_x_max = Slider(ax_x_max, 'Width', 0.1, 10.0, valinit=x_max_init, valstep=0.1)
    slider_y_max = Slider(ax_y_max, 'Height', 0.1, 10.0, valinit=y_max_init, valstep=0.1)
    slider_eccentricity = Slider(ax_eccentricity, 'Eccentricity', 0.1, 10.0, valinit=eccentricity_init, valstep=0.1)
    slider_c_factor = Slider(ax_c_factor, 'C Factor', 0.1, 5.0, valinit=c_factor_init, valstep=0.05)
    
    # Create radio buttons for function selection
    ax_func = plt.axes([0.025, 0.05, 0.15, 0.15])
    radio = RadioButtons(ax_func, list(function_dict.keys()), active=list(function_dict.keys()).index(current_function_name))
    
    def update(val):
        # Get current values from sliders
        r = slider_radius.val
        theta_x = slider_theta_x.val
        theta_y = slider_theta_y.val
        theta_z = slider_theta_z.val
        x_max = slider_x_max.val
        y_max = slider_y_max.val
        eccentricity = slider_eccentricity.val
        c_factor = slider_c_factor.val
        
        # Update rotation matrix
        rotation_matrix = create_rotation_matrix(theta_x, theta_y, theta_z)
        
        # Update circle radius
        circle1.set_radius(r)
        circle2.set_radius(r)
        circle3.set_radius(r)
        
        # Update grid (symmetric around zero)
        x = np.linspace(-x_max, x_max, resolution)
        y = np.linspace(-y_max, y_max, resolution)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Update original function display
        original_func_data = current_function(X, Y)
        original_im.set_array(original_func_data)
        original_im.set_extent([-x_max, x_max, -y_max, y_max])
        original_im.set_clim(original_func_data.min(), original_func_data.max())
        
        # Update direct warp data
        direct_warp_data = apply_direct_warp(
            current_function, x, y, r, rotation_matrix
        )
        
        # Update the direct warp plot
        direct_warp_im.set_array(direct_warp_data)
        direct_warp_im.set_extent([-x_max, x_max, -y_max, y_max])
        direct_warp_im.set_clim(direct_warp_data.min(), direct_warp_data.max())
        
        # Update full transformed data with eccentricity and c_factor
        transformed_data = apply_sequence_to_function(
            current_function, x, y, r, rotation_matrix, eccentricity, c_factor
        )
        
        # Update the transformed plot
        transformed_im.set_array(transformed_data)
        transformed_im.set_extent([-x_max, x_max, -y_max, y_max])
        transformed_im.set_clim(transformed_data.min(), transformed_data.max())
        
        # Update axis limits
        for ax in axs:
            ax.set_xlim(-x_max, x_max)
            ax.set_ylim(-y_max, y_max)
        
        fig.canvas.draw_idle()
    
    def update_function(function_name):
        nonlocal current_function, current_function_name
        current_function_name = function_name
        current_function = function_dict[function_name]
        
        # Get current values from sliders
        r = slider_radius.val
        theta_x = slider_theta_x.val
        theta_y = slider_theta_y.val
        theta_z = slider_theta_z.val
        x_max = slider_x_max.val
        y_max = slider_y_max.val
        eccentricity = slider_eccentricity.val
        c_factor = slider_c_factor.val
        
        # Update grid (symmetric around zero)
        x = np.linspace(-x_max, x_max, resolution)
        y = np.linspace(-y_max, y_max, resolution)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Update original function display
        original_func_data = current_function(X, Y)
        original_im.set_array(original_func_data)
        original_im.set_clim(original_func_data.min(), original_func_data.max())
        
        # Update rotation matrix
        rotation_matrix = create_rotation_matrix(theta_x, theta_y, theta_z)
        
        # Update direct warp data
        direct_warp_data = apply_direct_warp(
            current_function, x, y, r, rotation_matrix
        )
        
        # Update the direct warp plot
        direct_warp_im.set_array(direct_warp_data)
        direct_warp_im.set_clim(direct_warp_data.min(), direct_warp_data.max())
        
        # Update full transformed data with eccentricity and c_factor
        transformed_data = apply_sequence_to_function(
            current_function, x, y, r, rotation_matrix, eccentricity, c_factor
        )
        
        transformed_im.set_array(transformed_data)
        transformed_im.set_clim(transformed_data.min(), transformed_data.max())
        
        fig.canvas.draw_idle()
    
    # Register the update function with the sliders
    slider_radius.on_changed(update)
    slider_theta_x.on_changed(update)
    slider_theta_y.on_changed(update)
    slider_theta_z.on_changed(update)
    slider_x_max.on_changed(update)
    slider_y_max.on_changed(update)
    slider_eccentricity.on_changed(update)
    slider_c_factor.on_changed(update)
    
    # Register the function update callback
    radio.on_clicked(update_function)

    plt.show()

if __name__ == '__main__':
    main()
