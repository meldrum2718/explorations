
"""
Dynamic visualization for n-dimensional stereographic projections with animation.
Uses neural fields from neural_field.py and updates the fields with gradient descent
to minimize the difference between original and warped views.
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
from neural_field import NeuralField

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
    
    ### TODO update the code to just use a single field, dont have multiple fields
    field = NeuralField(seed=42, d_h=64, n_layers=3, L=6, output_dim=1, device=device)


    # field = create_random_fields(
    #     num_fields=num_fields, 
    #     seed=42, 
    #     output_dim=1,
    #     device=device
    # )
    
    # Create optimizers for each field
    optimizer = torch.optim.Adam(field.model.parameters(), lr=0.001)
    
    # # Define function dictionary with the neural fields
    # function_dict = {}
    # for i, field in enumerate(scalar_fields):
    #     function_dict[f'neural_field_{i+1}'] = field
    
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
    gradient_updates_enabled = False  # Flag to enable/disable gradient updates
    learning_rate_init = 0.001  # Initial learning rate
    steps_per_frame_init = 1  # Initial steps per frame
    
    
    # Create a simple figure with two subplots (original and warped)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Scaled coordinates for initial display
    X = normalized_X * domain_size_init
    Y = normalized_Y * domain_size_init
    
    # Original function display
    original_data = field(X, Y)
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
    
    # Add loss text display
    loss_text = fig.text(0.5, 0.95, 'Loss: N/A', ha='center', va='center', color='black')
    
    # Add circles to show boundary
    circle1 = plt.Circle((0, 0), r_init, fill=False, color='red', linestyle='--')
    circle2 = plt.Circle((0, 0), r_init, fill=False, color='red', linestyle='--')
    axs[0].add_patch(circle1)
    axs[1].add_patch(circle2)
    
    # Adjust the layout to make room for controls
    plt.subplots_adjust(bottom=0.4)
    
    # Add sliders for parameters
    ax_domain = plt.axes([0.25, 0.35, 0.65, 0.03])  # Domain size
    ax_radius = plt.axes([0.25, 0.30, 0.65, 0.03])  # Radius
    ax_theta_x = plt.axes([0.25, 0.25, 0.65, 0.03])  # X Rotation
    ax_theta_y = plt.axes([0.25, 0.20, 0.65, 0.03])  # Y Rotation
    ax_theta_z = plt.axes([0.25, 0.15, 0.65, 0.03])  # Z Rotation
    ax_anim_speed = plt.axes([0.25, 0.10, 0.65, 0.03])  # Animation Speed
    ax_learning_rate = plt.axes([0.25, 0.05, 0.65, 0.03])  # Learning Rate
    
    slider_domain = Slider(ax_domain, 'Domain Size', 0.1, 100.0, valinit=domain_size_init)
    slider_radius = Slider(ax_radius, 'Radius', 0.1, 10.0, valinit=r_init)
    slider_theta_x = Slider(ax_theta_x, 'X Rotation', -np.pi, np.pi, valinit=theta_x_init)
    slider_theta_y = Slider(ax_theta_y, 'Y Rotation', -np.pi, np.pi, valinit=theta_y_init)
    slider_theta_z = Slider(ax_theta_z, 'Z Rotation', -np.pi, np.pi, valinit=theta_z_init)
    slider_anim_speed = Slider(ax_anim_speed, 'Anim Speed', 0.1, 3.0, valinit=animation_speed_init)
    slider_learning_rate = Slider(ax_learning_rate, 'Learning Rate', 0.0001, 0.01, valinit=learning_rate_init)
    
    # Add animation control buttons
    ax_anim_toggle = plt.axes([0.05, 0.30, 0.15, 0.05])
    button_anim_toggle = Button(ax_anim_toggle, 'Start Animation')
    
    ax_grad_toggle = plt.axes([0.05, 0.20, 0.15, 0.05])
    button_grad_toggle = Button(ax_grad_toggle, 'Enable Gradients')
    
    ax_reset = plt.axes([0.05, 0.10, 0.15, 0.05])
    button_reset = Button(ax_reset, 'Reset Field')
    
    # Function to perform gradient updates on the neural field
    def perform_gradient_update(field, optimizer, original_coords, warped_coords, learning_rate):
        """
        Update the neural field to minimize the difference between evaluations
        at original and warped coordinates.
        
        Returns the loss value.
        """
        # Set learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Put models in training mode
        field.model.train()
        
        # Forward pass for original coordinates
        original_values = field.model(original_coords)
        
        # Forward pass for warped coordinates
        warped_values = field.model(warped_coords)
        
        # Compute loss (MSE between original and warped evaluations)
        loss = torch.nn.functional.mse_loss(original_values, warped_values)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Return the loss value
        return loss.item()
    
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
        field.model.eval()  # Set to eval mode for visualization
        original_data = field(X, Y)
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
        warped_data = field(warped_x, warped_y)
        
        # Update the warped image
        warped_im.set_array(warped_data)
        warped_im.set_extent([-domain_size, domain_size, -domain_size, domain_size])
        warped_im.set_clim(warped_data.min(), warped_data.max())
        
        # Update axis limits
        for ax in axs:
            ax.set_xlim(-domain_size, domain_size)
            ax.set_ylim(-domain_size, domain_size)
            
        return scaled_coords, warped_coords
    
    # This function will be called for each animation frame
    def animate(frame):
        if not animation_running:
            return [warped_im]
        
        # Get current parameter values - read directly from sliders
        domain_size = slider_domain.val
        r = slider_radius.val
        learning_rate = slider_learning_rate.val
        
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
        
        # Update both visualizations and get the coordinates
        original_coords, warped_coords = update_visualizations(domain_size, r, [theta_x, theta_y, theta_z])
        
        # Perform gradient update if enabled
        loss_val = 0
        if gradient_updates_enabled:
            # Perform a single step of gradient update
            loss_val = perform_gradient_update(
                field, 
                optimizer, 
                original_coords, 
                warped_coords, 
                learning_rate
            )
            
            # Update the visualizations again after the gradient update
            update_visualizations(domain_size, r, [theta_x, theta_y, theta_z])
            
            # Update the loss text
            loss_text.set_text(f'Loss: {loss_val:.6f}')
        
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
    
    # Function to update the neural field selection
    def update_function(function_name):
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
    
    # Function to toggle gradient updates
    def toggle_gradients(event):
        nonlocal gradient_updates_enabled
        gradient_updates_enabled = not gradient_updates_enabled
        
        if gradient_updates_enabled:
            button_grad_toggle.label.set_text('Disable Gradients')
            loss_text.set_text('Loss: 0.000000')
        else:
            button_grad_toggle.label.set_text('Enable Gradients')
            loss_text.set_text('Loss: N/A')
        
        fig.canvas.draw_idle()
    
    # Function to reset the neural field
    def reset_field(event):

        # Recreate the current field with the same parameters
        nonlocal field
        field = NeuralField(seed=42, d_h=64, n_layers=3, L=6, output_dim=1, device=device)
        
        # Create a new optimizer
        nonlocal optimizer
        optimizer = torch.optim.Adam(new_field.model.parameters(), lr=slider_learning_rate.val)
        
        # Force a full update
        update()
        
        # Update the loss text
        loss_text.set_text('Loss: N/A')
        
        fig.canvas.draw_idle()
    
    # Register event handlers
    slider_domain.on_changed(update)
    slider_radius.on_changed(update)
    slider_theta_x.on_changed(update)
    slider_theta_y.on_changed(update)
    slider_theta_z.on_changed(update)
    slider_learning_rate.on_changed(lambda val: None)  # No immediate update needed
    # radio.on_clicked(update_function)
    button_anim_toggle.on_clicked(toggle_animation)
    button_grad_toggle.on_clicked(toggle_gradients)
    button_reset.on_clicked(reset_field)
    
    # Initial update
    update()
    
    plt.tight_layout(rect=[0, 0.4, 1, 0.95])  # Adjust for the loss text at the top
    plt.show()

if __name__ == '__main__':
    main()
