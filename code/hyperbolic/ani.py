"""
Enhanced visualization tool for n-dimensional stereographic projections with animation support.
Uses the projection functions from projections.py for the technical implementation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
from matplotlib.animation import FuncAnimation

# Import the projection functions
from projections import (
    ball_to_flat,
    flat_to_ball,
    warp_with_rotation,
    create_nd_rotation_matrix
)

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
    rotation_matrix : torch.Tensor
        Rotation matrix for the n-sphere
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
    flat_points = ball_to_flat(points_torch, eccentricity, c_factor)
    warped_points = warp_with_rotation(flat_points, r, rotation_matrix)

    # Reshape back to grid
    tx = warped_points[:, 0].reshape(len(grid_x), len(grid_y)).numpy()
    ty = warped_points[:, 1].reshape(len(grid_x), len(grid_y)).numpy()
    
    # Evaluate function on transformed coordinates
    return function(tx, ty)

# Blend two images with a weight parameter
def blend_images(img1, img2, weight):
    """
    Linear interpolation between two images.
    
    Parameters:
    -----------
    img1, img2 : numpy.ndarray
        Images to blend
    weight : float
        Blend weight from 0 (img1 only) to 1 (img2 only)
    
    Returns:
    --------
    numpy.ndarray
        Blended image
    """
    return (1 - weight) * img1 + weight * img2

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
    
    # Animation parameters
    animation_speed_init = 2.0  # Cycles per second
    animation_running = False
    
    # For 2D points projected to 3D sphere, we need a 3x3 rotation matrix
    # Create initial rotation matrix with 3 angles (for 3D space)
    rotation_angles = [theta_x_init, theta_y_init, theta_z_init]
    rotation_matrix = create_nd_rotation_matrix(rotation_angles, 3)
    
    # Initial function choice
    current_function_name = 'complex_periodic'
    current_function = function_dict[current_function_name]
    
    # Create a figure with three subplots (original, direct warp, animation)
    fig, axs = plt.subplots(1, 3, figsize=(18, 7))
    
    # Original function display
    original_data = current_function(X, Y)
    original_im = axs[0].imshow(original_data, origin='lower', 
                               extent=[-x_max_init, x_max_init, -y_max_init, y_max_init],
                               cmap='viridis')
    axs[0].set_title('Original Function')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    fig.colorbar(original_im, ax=axs[0])
    
    # Full transformation function display (for middle panel)
    full_transform_data = apply_sequence_to_function(
        current_function, x, y, r_init, rotation_matrix, eccentricity_init, c_factor_init
    )
    full_transform_im = axs[1].imshow(full_transform_data, origin='lower', 
                                  extent=[-x_max_init, x_max_init, -y_max_init, y_max_init],
                                  cmap='viridis')
    axs[1].set_title('Full Transform (ball-to-flat → warp → flat-to-ball)')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    fig.colorbar(full_transform_im, ax=axs[1])
    
    # Full transformed function display (will be animated)
    transformed_data = apply_sequence_to_function(
        current_function, x, y, r_init, rotation_matrix, eccentricity_init, c_factor_init
    )
    
    # Initialize with original data (weight=0)
    blended_data = blend_images(original_data, transformed_data, 0)
    animated_im = axs[2].imshow(blended_data, origin='lower', 
                                extent=[-x_max_init, x_max_init, -y_max_init, y_max_init],
                                cmap='viridis')
    axs[2].set_title('Animated Blend (weight: 0.0)')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('y')
    fig.colorbar(animated_im, ax=axs[2])
    
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
    ax_weight = plt.axes([0.25, 0.05, 0.65, 0.03])  # Manual weight control
    
    slider_radius = Slider(ax_radius, 'Radius', 0.005, 100.0, valinit=r_init)
    slider_theta_x = Slider(ax_theta_x, 'Z Rotation', -np.pi, np.pi, valinit=theta_x_init, valstep=0.05)
    slider_theta_y = Slider(ax_theta_y, 'Y Rotation', -np.pi, np.pi, valinit=theta_y_init, valstep=0.05)
    slider_theta_z = Slider(ax_theta_z, 'X Rotation', -np.pi, np.pi, valinit=theta_z_init, valstep=0.05)
    slider_x_max = Slider(ax_x_max, 'Width', 0.1, 10.0, valinit=x_max_init, valstep=0.1)
    slider_y_max = Slider(ax_y_max, 'Height', 0.1, 10.0, valinit=y_max_init, valstep=0.1)
    slider_eccentricity = Slider(ax_eccentricity, 'Eccentricity', 0.1, 10.0, valinit=eccentricity_init, valstep=0.1)
    slider_c_factor = Slider(ax_c_factor, 'C Factor', 0.1, 5.0, valinit=c_factor_init, valstep=0.05)
    slider_weight = Slider(ax_weight, 'Stereo. Weight', 0.0, 1.0, valinit=0.0, valstep=0.01)
    
    # Add animation control buttons
    ax_anim_speed = plt.axes([0.05, 0.48, 0.15, 0.03])
    slider_anim_speed = Slider(ax_anim_speed, 'Anim Speed', 0.1, 5.0, valinit=animation_speed_init, valstep=0.1)
    
    ax_anim_toggle = plt.axes([0.05, 0.40, 0.15, 0.05])
    button_anim_toggle = Button(ax_anim_toggle, 'Start Animation')
    
    # Create radio buttons for function selection
    ax_func = plt.axes([0.025, 0.05, 0.15, 0.15])
    radio = RadioButtons(ax_func, list(function_dict.keys()), active=list(function_dict.keys()).index(current_function_name))
    
    # Animation function
    def animate(frame):
        nonlocal animation_running
        
        if not animation_running:
            return [animated_im]
        
        # Calculate weight based on frame (sinusoidal oscillation)
        weight = 0.5 * (1 + np.sin(2 * np.pi * slider_anim_speed.val * frame / 100))
        slider_weight.set_val(weight)
        
        # No need to update anything here as slider_weight.set_val will trigger update()
        return [animated_im]
    
    # Initialize the animation object
    anim = FuncAnimation(fig, animate, frames=1000, interval=20, blit=True)
    anim.pause()  # Start paused
    
    def update(val=None):
        # Get current values from sliders
        r = slider_radius.val
        theta_z = slider_theta_x.val  # This is Z rotation (slider_theta_x has Z label)
        theta_y = slider_theta_y.val
        theta_x = slider_theta_z.val  # This is X rotation (slider_theta_z has X label)
        x_max = slider_x_max.val
        y_max = slider_y_max.val
        eccentricity = slider_eccentricity.val
        c_factor = slider_c_factor.val
        weight = slider_weight.val
        
        # Update rotation matrix
        rotation_angles = [theta_x, theta_y, theta_z]
        rotation_matrix = create_nd_rotation_matrix(rotation_angles, 3)
        
        # Update circle radius
        circle1.set_radius(r)
        circle2.set_radius(r)
        circle3.set_radius(r)
        
        # Update grid (symmetric around zero)
        x = np.linspace(-x_max, x_max, resolution)
        y = np.linspace(-y_max, y_max, resolution)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Update original function display
        original_data = current_function(X, Y)
        original_im.set_array(original_data)
        original_im.set_extent([-x_max, x_max, -y_max, y_max])
        original_im.set_clim(original_data.min(), original_data.max())
        
        # Update full transform data
        full_transform_data = apply_sequence_to_function(
            current_function, x, y, r, rotation_matrix, eccentricity, c_factor
        )
        
        # Update the full transform plot
        full_transform_im.set_array(full_transform_data)
        full_transform_im.set_extent([-x_max, x_max, -y_max, y_max])
        full_transform_im.set_clim(full_transform_data.min(), full_transform_data.max())
        
        # Update full transformed data with eccentricity and c_factor
        transformed_data = apply_sequence_to_function(
            current_function, x, y, r, rotation_matrix, eccentricity, c_factor
        )
        
        # Create blended image based on weight
        blended_data = blend_images(original_data, transformed_data, weight)
        
        # Update the animated plot
        animated_im.set_array(blended_data)
        animated_im.set_extent([-x_max, x_max, -y_max, y_max])
        animated_im.set_clim(
            min(original_data.min(), transformed_data.min()),
            max(original_data.max(), transformed_data.max())
        )
        
        # Update axis limits
        for ax in axs:
            ax.set_xlim(-x_max, x_max)
            ax.set_ylim(-y_max, y_max)
        
        # Update the title with current weight
        axs[2].set_title(f'Animated Blend (weight: {weight:.2f})')
        
        fig.canvas.draw_idle()
    
    def update_function(function_name):
        nonlocal current_function, current_function_name
        current_function_name = function_name
        current_function = function_dict[function_name]
        update()
    
    def toggle_animation(event):
        nonlocal animation_running
        animation_running = not animation_running
        
        if animation_running:
            button_anim_toggle.label.set_text('Stop Animation')
            anim.resume()
        else:
            button_anim_toggle.label.set_text('Start Animation')
            anim.pause()
        
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
    slider_weight.on_changed(update)
    
    # Register animation controls
    button_anim_toggle.on_clicked(toggle_animation)
    
    # Register the function update callback
    radio.on_clicked(update_function)

    plt.show()

if __name__ == '__main__':
    main()
