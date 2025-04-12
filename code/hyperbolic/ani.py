"""
Simplified visualization for n-dimensional stereographic projections with animation.
Uses neural fields from neural_field.py and focuses on scalar fields only.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.animation import FuncAnimation

# Import the projection functions
from projections import (
    ball_to_flat,
    flat_to_ball,
    warp_with_rotation,
    create_nd_rotation_matrix
)

# Import neural field
from neural_field import create_random_fields, NeuralField

def apply_warp_to_coords(coords, r, rotation_angles, eccentricity, c_factor):
    """
    Apply the warping sequence to input coordinates.
    
    Parameters:
    -----------
    coords : torch.Tensor
        Input coordinates, shape (B, 2)
    r : float
        Radius parameter
    rotation_angles : list or tensor
        Rotation angles to create rotation matrix
    eccentricity : float
        Scaling factor for sphere coordinates
    c_factor : float
        Factor to scale the center point
    
    Returns:
    --------
    torch.Tensor
        Transformed coordinates
    """
    device = coords.device
    
    # Create rotation matrix on the same device as coords
    rotation_matrix = create_nd_rotation_matrix(rotation_angles, 3).to(device)
    
    flat_points = ball_to_flat(coords, eccentricity, c_factor)
    warped_points = warp_with_rotation(flat_points, r, rotation_matrix)
    return warped_points

def main():
    # Set device for torch
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 
                          'cpu')
    print(f"Using device: {device}")
    
    # Create random neural fields (scalar only)
    num_fields = 3
    scalar_fields = create_random_fields(
        num_fields=num_fields, 
        seed=42, 
        output_dim=1,
        device=device
    )
    
    # Define function dictionary with the neural fields
    function_dict = {}
    for i, field in enumerate(scalar_fields):
        function_dict[f'neural_field_{i+1}'] = field
    
    # Set up grid parameters
    resolution = 200  # Reduced for faster rendering
    domain_size_init = 2.0  # Initial domain size (half-width and half-height)
    r_init = 2.0
    eccentricity_init = 3.0
    c_factor_init = 1.34
    
    # Create normalized grid (fixed from -1 to 1)
    # This grid is fixed and will be scaled by domain_size when needed
    normalized_x = np.linspace(-1, 1, resolution)
    normalized_y = np.linspace(-1, 1, resolution)
    normalized_X, normalized_Y = np.meshgrid(normalized_x, normalized_y, indexing='ij')
    
    # Create normalized coordinates tensor (fixed)
    normalized_grid_coords = np.stack([normalized_X.flatten(), normalized_Y.flatten()], axis=-1)
    normalized_coords_torch = torch.tensor(normalized_grid_coords, dtype=torch.float32, device=device)
    
    # Initial rotation angles (in radians)
    theta_x_init = 0.0
    theta_y_init = 0.0
    theta_z_init = 0.0
    
    # Animation parameters
    animation_speed_init = 1.0
    animation_running = False
    
    # Initial function choice
    current_function_name = list(function_dict.keys())[0]
    current_function = function_dict[current_function_name]
    
    # Create a simple figure with two subplots (original and warped)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Scaled coordinates for initial display
    X = normalized_X * domain_size_init
    Y = normalized_Y * domain_size_init
    
    # Original function display
    original_data = current_function(X, Y)
    original_im = axs[0].imshow(original_data, origin='lower', 
                              extent=[-domain_size_init, domain_size_init, -domain_size_init, domain_size_init],
                              cmap='viridis')
    axs[0].set_title('Original Neural Field')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    fig.colorbar(original_im, ax=axs[0])
    
    # Warped function display (will be updated in animation)
    # Initially just show the original
    warped_im = axs[1].imshow(original_data, origin='lower', 
                            extent=[-domain_size_init, domain_size_init, -domain_size_init, domain_size_init],
                            cmap='viridis')
    axs[1].set_title('Warped Neural Field')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    fig.colorbar(warped_im, ax=axs[1])
    
    # Add circles to show boundary
    circle1 = plt.Circle((0, 0), r_init, fill=False, color='red', linestyle='--')
    circle2 = plt.Circle((0, 0), r_init, fill=False, color='red', linestyle='--')
    axs[0].add_patch(circle1)
    axs[1].add_patch(circle2)
    
    # Adjust the layout to make room for controls
    plt.subplots_adjust(bottom=0.35)
    
    # Add sliders for parameters
    ax_domain = plt.axes([0.25, 0.30, 0.65, 0.03])  # New slider for domain size
    ax_radius = plt.axes([0.25, 0.25, 0.65, 0.03])
    ax_theta_x = plt.axes([0.25, 0.20, 0.65, 0.03])
    ax_theta_y = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_theta_z = plt.axes([0.25, 0.10, 0.65, 0.03])
    ax_anim_speed = plt.axes([0.25, 0.05, 0.65, 0.03])
    
    slider_domain = Slider(ax_domain, 'Domain Size', 0.1, 50.0, valinit=domain_size_init)
    slider_radius = Slider(ax_radius, 'Radius', 0.1, 10.0, valinit=r_init)
    slider_theta_x = Slider(ax_theta_x, 'X Rotation', -np.pi, np.pi, valinit=theta_x_init)
    slider_theta_y = Slider(ax_theta_y, 'Y Rotation', -np.pi, np.pi, valinit=theta_y_init)
    slider_theta_z = Slider(ax_theta_z, 'Z Rotation', -np.pi, np.pi, valinit=theta_z_init)
    slider_anim_speed = Slider(ax_anim_speed, 'Anim Speed', 0.1, 3.0, valinit=animation_speed_init)
    
    # Add animation control button
    ax_anim_toggle = plt.axes([0.05, 0.25, 0.15, 0.05])
    button_anim_toggle = Button(ax_anim_toggle, 'Start Animation')
    
    # Create radio buttons for function selection
    ax_func = plt.axes([0.05, 0.05, 0.15, 0.15])
    radio = RadioButtons(ax_func, list(function_dict.keys()), active=list(function_dict.keys()).index(current_function_name))
    
    # Function to update both original and warped visualizations
    def update_visualizations(domain_size, r, rotation_angles):
        """Update both original and warped visualizations with current parameters"""
        # Update circle radius
        circle1.set_radius(r)
        circle2.set_radius(r)
        
        # Scale the normalized grid by the domain size for original view
        X = normalized_X * domain_size
        Y = normalized_Y * domain_size
        
        # Update the original image
        original_data = current_function(X, Y)
        original_im.set_array(original_data)
        original_im.set_extent([-domain_size, domain_size, -domain_size, domain_size])
        original_im.set_clim(original_data.min(), original_data.max())
        
        # Scale the normalized coordinates by the domain size for warped view
        scaled_coords = normalized_coords_torch * domain_size
        
        # Apply warping to the coordinates
        warped_coords = apply_warp_to_coords(
            scaled_coords, r, rotation_angles, eccentricity_init, c_factor_init
        )
        
        # Reshape coordinates for the neural field
        warped_x = warped_coords[:, 0].reshape(resolution, resolution).cpu().numpy()
        warped_y = warped_coords[:, 1].reshape(resolution, resolution).cpu().numpy()
        
        # Evaluate the neural field on the warped coordinates
        warped_data = current_function(warped_x, warped_y)
        
        # Update the warped image
        warped_im.set_array(warped_data)
        warped_im.set_extent([-domain_size, domain_size, -domain_size, domain_size])
        warped_im.set_clim(warped_data.min(), warped_data.max())
        
        # Update axis limits
        for ax in axs:
            ax.set_xlim(-domain_size, domain_size)
            ax.set_ylim(-domain_size, domain_size)
    
    # This function will be called for each animation frame
    def animate(frame):
        if not animation_running:
            return [warped_im]
        
        # Get current parameter values - read directly from sliders
        domain_size = slider_domain.val
        r = slider_radius.val
        
        # Calculate rotation angles based on frame and animation speed
        speed = slider_anim_speed.val
        theta_x = slider_theta_x.val + 0.05 * speed * np.sin(0.01 * frame)
        theta_y = slider_theta_y.val + 0.05 * speed * np.sin(0.02 * frame)
        theta_z = slider_theta_z.val + 0.05 * speed * np.cos(0.015 * frame)
        
        # Update sliders to show current values (without triggering their callbacks)
        slider_theta_x.eventson = False
        slider_theta_y.eventson = False
        slider_theta_z.eventson = False
        
        slider_theta_x.set_val(theta_x)
        slider_theta_y.set_val(theta_y)
        slider_theta_z.set_val(theta_z)
        
        slider_theta_x.eventson = True
        slider_theta_y.eventson = True
        slider_theta_z.eventson = True
        
        # Update both visualizations
        update_visualizations(domain_size, r, [theta_x, theta_y, theta_z])
        
        return [warped_im]
    
    # Initialize the animation object
    anim = FuncAnimation(fig, animate, frames=1000, interval=30, blit=True)
    anim.pause()  # Start paused
    
    # Function to update display when sliders change
    def update(val=None):
        if not animation_running:  # Only update manually if animation is not running
            # Get current parameter values
            domain_size = slider_domain.val
            r = slider_radius.val
            theta_x = slider_theta_x.val
            theta_y = slider_theta_y.val
            theta_z = slider_theta_z.val
            
            # Update both visualizations
            update_visualizations(domain_size, r, [theta_x, theta_y, theta_z])
            
            fig.canvas.draw_idle()
    
    # Function to update the neural field
    def update_function(function_name):
        nonlocal current_function, current_function_name
        current_function_name = function_name
        current_function = function_dict[function_name]
        
        # Force a full update to refresh everything
        update()
    
    # Function to toggle animation
    def toggle_animation(event):
        nonlocal animation_running
        animation_running = not animation_running
        
        if animation_running:
            button_anim_toggle.label.set_text('Stop Animation')
            anim.resume()
        else:
            button_anim_toggle.label.set_text('Start Animation')
            anim.pause()
            # Force an update to ensure display is correct
            update()
        
        fig.canvas.draw_idle()
    
    # Register event handlers
    slider_domain.on_changed(update)
    slider_radius.on_changed(update)
    slider_theta_x.on_changed(update)
    slider_theta_y.on_changed(update)
    slider_theta_z.on_changed(update)
    radio.on_clicked(update_function)
    button_anim_toggle.on_clicked(toggle_animation)
    
    # Initial update
    update()
    
    plt.tight_layout(rect=[0, 0.35, 1, 1])
    plt.show()

if __name__ == '__main__':
    main()
