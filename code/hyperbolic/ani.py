"""
Dynamic visualization for n-dimensional stereographic projections with animation.
Uses a tensor-based approach with coordinate transformations between ball and flat spaces,
with state evolution through time.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
import torch
from scipy.interpolate import LinearNDInterpolator

# Import the projection functions
from projections import (
    ball_to_flat,
    flat_to_ball,
    warp_with_rotation,
    create_nd_rotation_matrix
)

def create_function_tensor(resolution=200, domain_size=2.0, device='cpu'):
    """
    Create a tensor representing a function on a grid.
    The function is defined on the square [-domain_size, domain_size]^2
    and zeroed out outside the circle with radius domain_size.
    
    Parameters:
    -----------
    resolution : int
        Grid resolution
    domain_size : float
        Size of the domain (half-width and half-height)
    device : torch.device
        Device to put the tensor on
        
    Returns:
    --------
    function_tensor : torch.Tensor
        2D tensor containing function values
    grid_coords : torch.Tensor
        Tensor of shape (resolution^2, 2) containing grid coordinates
    """
    with torch.no_grad():  # Disable gradient tracking
        # Create grid coordinates
        x = torch.linspace(-domain_size, domain_size, resolution)
        y = torch.linspace(-domain_size, domain_size, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Create coordinates tensor
        grid_coords = torch.stack([X.flatten(), Y.flatten()], dim=-1).to(device)
        
        # Calculate the norm of each coordinate
        norms = torch.sqrt(grid_coords[:, 0]**2 + grid_coords[:, 1]**2)
        
        # Calculate function values - use example function sin(xy) + cos(2y) + sin(3x)
        values = torch.sin(grid_coords[:, 0] * grid_coords[:, 1]) + \
                torch.cos(2 * grid_coords[:, 1]) + \
                torch.sin(3 * grid_coords[:, 0])
        
        # Zero out values outside the circle
        mask = norms < domain_size
        values = values * mask
        
        # Reshape to 2D tensor
        function_tensor = values.reshape(resolution, resolution)
        
    return function_tensor, grid_coords

def sample_function_with_interpolation(coords, function_tensor, grid_coords, resolution):
    """
    Sample the function at given coordinates using linear interpolation.
    
    Parameters:
    -----------
    coords : torch.Tensor
        Coordinates to sample at, shape (N, 2)
    function_tensor : torch.Tensor
        2D tensor containing function values
    grid_coords : torch.Tensor
        Tensor of shape (resolution^2, 2) containing grid coordinates
    resolution : int
        Grid resolution
        
    Returns:
    --------
    sampled_values : torch.Tensor
        Function values at the given coordinates
    """
    # Convert tensors to numpy for interpolation - detach to ensure no gradient tracking
    coords_np = coords.detach().cpu().numpy()
    grid_coords_np = grid_coords.detach().cpu().numpy()
    values_np = function_tensor.detach().flatten().cpu().numpy()
    
    # Create interpolator - this doesn't maintain any gradient information
    interpolator = LinearNDInterpolator(grid_coords_np, values_np, fill_value=0)
    
    # Sample function
    sampled_values = interpolator(coords_np)
    
    # Convert back to tensor but don't require gradients
    return torch.tensor(sampled_values, dtype=torch.float32, device=coords.device, requires_grad=False)

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

def main():
    # Explicitly disable gradient computation since we don't need it
    torch.set_grad_enabled(False)
    
    # Set device for torch
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 
                          'cpu')
    print(f"Using device: {device}")
    
    # Set up grid parameters
    resolution = 200  # Grid resolution
    
    domain_size_init = 2.0  # Initial domain size (half-width and half-height)
    r_init = 2.0
    eccentricity_init = 3.0
    c_factor_init = 1.34
    
    # Create function tensor and grid coordinates
    function_tensor, grid_coords = create_function_tensor(resolution, domain_size_init, device)
    
    # Create a copy of the original tensor for reference and resets
    original_tensor = function_tensor.clone()
    
    # Create normalized grid for visualization
    normalized_x = np.linspace(-1, 1, resolution)
    normalized_y = np.linspace(-1, 1, resolution)
    normalized_X, normalized_Y = np.meshgrid(normalized_x, normalized_y, indexing='ij')
    
    # Create normalized coordinates tensor
    normalized_grid_coords = np.stack([normalized_X.flatten(), normalized_Y.flatten()], axis=-1)
    normalized_coords_torch = torch.tensor(normalized_grid_coords, dtype=torch.float32, device=device)
    
    # Initial rotation angles (in radians)
    theta_x_init = 0.0
    theta_y_init = 0.0
    theta_z_init = 0.0
    
    # Animation parameters
    animation_speed_init = 1.0
    animation_running = False
    
    # Animation angle offsets - these will be used to calculate actual angles during animation
    # without updating the sliders
    theta_x_offset = 0.0
    theta_y_offset = 0.0
    theta_z_offset = 0.0
    
    # Create a simple figure with two subplots (original and blended)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original function display - now shows the current state
    original_im = axs[0].imshow(function_tensor.cpu().numpy(), origin='lower', 
                              extent=[-domain_size_init, domain_size_init, -domain_size_init, domain_size_init],
                              cmap='viridis')
    axs[0].set_title('Current State')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    fig.colorbar(original_im, ax=axs[0])
    
    # Second panel shows the warped state (before blending)
    warped_im = axs[1].imshow(function_tensor.cpu().numpy(), origin='lower', 
                            extent=[-domain_size_init, domain_size_init, -domain_size_init, domain_size_init],
                            cmap='viridis')
    axs[1].set_title('Warped State')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    fig.colorbar(warped_im, ax=axs[1])
    
    # Set up titles and explanatory text
    fig.text(0.5, 0.97, 'Tensor-Based Stereographic Projection with State Evolution', 
             ha='center', va='center', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.93, 'Left: Current state | Right: Warped state (before blending)', 
             ha='center', va='center', fontsize=10)
    
    # Add rotation angle display
    rotation_text = fig.text(0.5, 0.9, 'Rotation: [0.00, 0.00, 0.00]', ha='center', va='center', color='black')
    
    # Add circles to show boundary
    circle1 = plt.Circle((0, 0), domain_size_init, fill=False, color='red', linestyle='--')
    circle2 = plt.Circle((0, 0), domain_size_init, fill=False, color='red', linestyle='--')
    axs[0].add_patch(circle1)
    axs[1].add_patch(circle2)
    
    # Set up blend factor display
    blend_text = fig.text(0.5, 0.85, 'Blend Factor: 0.50', ha='center', va='center', color='black')
    
    # Adjust the layout to make room for controls
    plt.subplots_adjust(bottom=0.35)
    
    # Add sliders for parameters
    ax_domain = plt.axes([0.25, 0.30, 0.65, 0.03])  # Domain size
    ax_radius = plt.axes([0.25, 0.25, 0.65, 0.03])  # Radius
    ax_theta_x = plt.axes([0.25, 0.20, 0.65, 0.03])  # X Rotation
    ax_theta_y = plt.axes([0.25, 0.15, 0.65, 0.03])  # Y Rotation
    ax_theta_z = plt.axes([0.25, 0.10, 0.65, 0.03])  # Z Rotation
    ax_anim_speed = plt.axes([0.25, 0.05, 0.65, 0.03])  # Animation Speed
    ax_blend = plt.axes([0.25, 0.00, 0.65, 0.03])  # Blending factor
    
    slider_domain = Slider(ax_domain, 'Domain Size', 0.1, 10.0, valinit=domain_size_init)
    slider_radius = Slider(ax_radius, 'Radius', 0.1, 10.0, valinit=r_init)
    slider_theta_x = Slider(ax_theta_x, 'X Rotation', -np.pi, np.pi, valinit=theta_x_init)
    slider_theta_y = Slider(ax_theta_y, 'Y Rotation', -np.pi, np.pi, valinit=theta_y_init)
    slider_theta_z = Slider(ax_theta_z, 'Z Rotation', -np.pi, np.pi, valinit=theta_z_init)
    slider_anim_speed = Slider(ax_anim_speed, 'Anim Speed', 0.1, 3.0, valinit=animation_speed_init)
    slider_blend = Slider(ax_blend, 'Blend Factor', 0.0, 1.0, valinit=0.5)
    
    # Add button to regenerate function tensor
    ax_regen = plt.axes([0.05, 0.25, 0.15, 0.05])
    button_regen = Button(ax_regen, 'Regenerate Function')
    
    # Add animation control button
    ax_anim_toggle = plt.axes([0.05, 0.15, 0.15, 0.05])
    button_anim_toggle = Button(ax_anim_toggle, 'Start Animation')
    
    # Add button to reset state to original
    ax_reset_state = plt.axes([0.05, 0.05, 0.15, 0.05])
    button_reset_state = Button(ax_reset_state, 'Reset State')
    
    # Function to update both panels with current and warped states
    def update_visualizations(domain_size, r, rotation_angles):
        """Update visualizations with current state and warped state"""
        nonlocal function_tensor, grid_coords
        
        # Update circle radius
        circle1.set_radius(domain_size)
        circle2.set_radius(domain_size)
        
        # Scale the normalized coordinates by the domain size
        scaled_coords = normalized_coords_torch * domain_size
        
        # For visualization, we do the inverse process:
        # 1. For each output coordinate, find where it came from in the input
        # 2. Sample the function at those input coordinates
        
        # Apply warping to the coordinates
        with torch.no_grad():  # Disable gradient tracking
            warped_coords = apply_warp_to_coords(
                scaled_coords, r, rotation_angles, eccentricity_init, c_factor_init
            )
            
            # To sample the function, we need to do the inverse transform:
            # Push forward the coordinates from flat space to ball space
            inverse_coords = flat_to_ball(warped_coords, eccentricity_init, c_factor_init)
        
        # Sample the function at the inverse coordinates
        with torch.no_grad():  # Disable gradient tracking
            warped_values = sample_function_with_interpolation(
                inverse_coords, function_tensor, grid_coords, resolution
            )
            
            # Reshape for visualization
            warped_data = warped_values.reshape(resolution, resolution)
            
            # Get blend factor
            blend_factor = slider_blend.val
            
            # First display the current state
            original_im.set_array(function_tensor.cpu().numpy())
            
            # Display the warped state before blending
            warped_im.set_array(warped_data.cpu().numpy())
            
            # Now update the actual function tensor with the blend
            function_tensor = (1 - blend_factor) * function_tensor + blend_factor * warped_data
        
        # Update color scales
        original_im.set_clim(function_tensor.min().item(), function_tensor.max().item())
        warped_im.set_clim(warped_data.min().item(), warped_data.max().item())
        
        # Update axis limits
        for ax in axs:
            ax.set_xlim(-domain_size, domain_size)
            ax.set_ylim(-domain_size, domain_size)
    
    # Function to regenerate the function tensor
    def regenerate_function(event=None):
        nonlocal function_tensor, grid_coords, original_tensor
        domain_size = slider_domain.val
        
        # Free memory by explicitly deleting old tensors
        del function_tensor
        del original_tensor
        
        # Create new function tensor
        function_tensor, grid_coords = create_function_tensor(resolution, domain_size, device)
        
        # Update original tensor reference
        with torch.no_grad():  # Disable gradient tracking
            original_tensor = function_tensor.clone()
        
        # Update both displays
        original_im.set_array(function_tensor.cpu().numpy())
        original_im.set_extent([-domain_size, domain_size, -domain_size, domain_size])
        original_im.set_clim(function_tensor.min().item(), function_tensor.max().item())
        
        # Set warped display to also show the initial state initially
        warped_im.set_array(function_tensor.cpu().numpy())
        warped_im.set_extent([-domain_size, domain_size, -domain_size, domain_size])
        warped_im.set_clim(function_tensor.min().item(), function_tensor.max().item())
        
        # Clear the figure cache to reduce memory usage
        fig.canvas.flush_events()
        fig.canvas.draw_idle()
    
    # Function to reset the state to the original function
    def reset_state(event=None):
        nonlocal function_tensor, original_tensor
        with torch.no_grad():  # Disable gradient tracking
            function_tensor = original_tensor.clone()
            
        # Update the display - now shows current state in left panel
        original_im.set_array(function_tensor.cpu().numpy())
        original_im.set_clim(function_tensor.min().item(), function_tensor.max().item())
        
        # Warped state will be updated on the next animation frame or slider change
        
        # Clear the figure cache to reduce memory usage
        fig.canvas.flush_events()
        fig.canvas.draw_idle()
        
    # Add a frame counter to limit updates and prevent slowdown
    frame_counter = 0
    update_frequency = 2  # Update every N frames
    
    # This function will be called for each animation frame
    def animate(frame):
        if not animation_running:
            return [warped_im]
        
        # Use a frame counter to reduce update frequency and prevent slowdown
        nonlocal frame_counter
        frame_counter += 1
        if frame_counter % update_frequency != 0:
            return [warped_im]
        
        # Get base parameter values from sliders
        domain_size = slider_domain.val
        r = slider_radius.val
        base_theta_x = slider_theta_x.val
        base_theta_y = slider_theta_y.val
        base_theta_z = slider_theta_z.val
        speed = slider_anim_speed.val
        blend_factor = slider_blend.val
        
        # Calculate rotation angles based on frame and animation speed
        nonlocal theta_x_offset, theta_y_offset, theta_z_offset
        theta_x_offset = 0.05 * speed * np.sin(0.01 * frame)
        theta_y_offset = 0.05 * speed * np.sin(0.02 * frame)
        theta_z_offset = 0.05 * speed * np.cos(0.015 * frame)
        
        # Calculate actual angles for this frame
        theta_x = base_theta_x + theta_x_offset
        theta_y = base_theta_y + theta_y_offset
        theta_z = base_theta_z + theta_z_offset
        
        # Update the rotation text display 
        rotation_text.set_text(f'Rotation: [{theta_x:.2f}, {theta_y:.2f}, {theta_z:.2f}]')
        blend_text.set_text(f'Blend Factor: {blend_factor:.2f}')
        
        # Update both visualizations
        update_visualizations(domain_size, r, [theta_x, theta_y, theta_z])
        
        # Explicitly clear unused objects for garbage collection
        import gc
        gc.collect()
        
        return [warped_im]
    
    # Initialize the animation object with save_count=1 to prevent memory growth
    anim = FuncAnimation(fig, animate, frames=1000, interval=30, blit=True, save_count=1)
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
            blend_factor = slider_blend.val
            
            # Update blend factor display
            blend_text.set_text(f'Blend Factor: {blend_factor:.2f}')
            
            # Check if domain size changed, if so regenerate function tensor
            if val is slider_domain:
                regenerate_function()
                return
            
            # Reset rotation offsets when manually updating
            nonlocal theta_x_offset, theta_y_offset, theta_z_offset
            theta_x_offset = 0.0
            theta_y_offset = 0.0
            theta_z_offset = 0.0
            
            # Update the rotation text
            rotation_text.set_text(f'Rotation: [{theta_x:.2f}, {theta_y:.2f}, {theta_z:.2f}]')
            
            # Update visualizations
            update_visualizations(domain_size, r, [theta_x, theta_y, theta_z])
            
            fig.canvas.draw_idle()
    
    # Function to toggle animation
    def toggle_animation(event):
        nonlocal animation_running
        animation_running = not animation_running
        
        if animation_running:
            button_anim_toggle.label.set_text('Stop Animation')
            # Reset rotation offsets when starting animation
            nonlocal theta_x_offset, theta_y_offset, theta_z_offset
            theta_x_offset = 0.0
            theta_y_offset = 0.0
            theta_z_offset = 0.0
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
    slider_anim_speed.on_changed(update)
    slider_blend.on_changed(update)  
    button_anim_toggle.on_clicked(toggle_animation)
    button_regen.on_clicked(regenerate_function)
    button_reset_state.on_clicked(reset_state)
    
    # Initial update
    update()
    
    plt.tight_layout(rect=[0, 0.35, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    main()
