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

def create_function_tensor(resolution=200, domain_size=2.0, eccentricity=3.0, c_factor=1.34, device='cpu'):
    """
    Create a tensor representing a function on a grid.
    The function is defined on the square [-domain_size, domain_size]^2
    and zeroed out outside the circle with radius domain_size.
    Uses ball_to_flat transformation to create the grid in flat space.
    
    Parameters:
    -----------
    resolution : int
        Grid resolution
    domain_size : float
        Size of the domain (half-width and half-height)
    eccentricity : float
        Eccentricity parameter for ball_to_flat transformation
    c_factor : float
        C-factor parameter for ball_to_flat transformation
    device : torch.device
        Device to put the tensor on
        
    Returns:
    --------
    function_tensor : torch.Tensor
        2D tensor containing function values
    ball_coords : torch.Tensor
        Tensor of shape (resolution^2, 2) containing ball space coordinates
    flat_coords : torch.Tensor
        Tensor of shape (resolution^2, 2) containing flat space coordinates
    """
    with torch.no_grad():  # Disable gradient tracking
        # Create grid coordinates in ball space
        x = torch.linspace(-domain_size, domain_size, resolution)
        y = torch.linspace(-domain_size, domain_size, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Create coordinates tensor for ball space
        ball_coords = torch.stack([X.flatten(), Y.flatten()], dim=-1).to(device)
        
        # Calculate the norm of each coordinate
        norms = torch.sqrt(ball_coords[:, 0]**2 + ball_coords[:, 1]**2)
        
        # Convert ball space coordinates to flat space using ball_to_flat
        flat_coords = ball_to_flat(ball_coords, eccentricity, c_factor)
        
        # Calculate function values using the flat space coordinates
        # We use the same function but apply it to flat space coordinates
        values = torch.sin(flat_coords[:, 0] * flat_coords[:, 1]) + \
                torch.cos(2 * flat_coords[:, 1]) + \
                torch.sin(3 * flat_coords[:, 0])
        
        # Zero out values outside the circle in ball space
        mask = norms < domain_size
        values = values * mask
        
        # Reshape to 2D tensor
        function_tensor = values.reshape(resolution, resolution)
        
    return function_tensor, ball_coords, flat_coords

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
    
    # Default parameters for the left plot (original state)
    left_eccentricity = 2.0
    left_c_factor = 1.0
    
    # Parameters for the right plot (warped state) - these will be adjustable via sliders
    eccentricity_init = 3.0
    c_factor_init = 1.34
    
    # Create function tensors for both plots with their respective parameters
    # Left plot (original state)
    left_function_tensor, left_ball_coords, left_flat_coords = create_function_tensor(
        resolution, domain_size_init, left_eccentricity, left_c_factor, device
    )
    
    # Right plot (warped state)
    right_function_tensor, right_ball_coords, right_flat_coords = create_function_tensor(
        resolution, domain_size_init, eccentricity_init, c_factor_init, device
    )
    
    # Create copies of the original tensors for reference and resets
    original_left_tensor = left_function_tensor.clone()
    original_left_ball_coords = left_ball_coords.clone()
    original_left_flat_coords = left_flat_coords.clone()
    
    original_right_tensor = right_function_tensor.clone()
    original_right_ball_coords = right_ball_coords.clone()
    original_right_flat_coords = right_flat_coords.clone()
    
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
    
    # Normalization options
    auto_normalize = False  # Whether to apply normalization automatically at each step
    
    # Animation angle offsets - these will be used to calculate actual angles during animation
    # without updating the sliders
    theta_x_offset = 0.0
    theta_y_offset = 0.0
    theta_z_offset = 0.0
    
    # Create a simple figure with two subplots (original and blended)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left plot - displays the original state with fixed projection parameters
    left_im = axs[0].imshow(left_function_tensor.cpu().numpy(), origin='lower', 
                              extent=[-domain_size_init, domain_size_init, -domain_size_init, domain_size_init],
                              cmap='viridis')
    axs[0].set_title(f'Original State (e={left_eccentricity}, c={left_c_factor})')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    fig.colorbar(left_im, ax=axs[0])
    
    # Right plot - shows the warped state with adjustable parameters
    right_im = axs[1].imshow(right_function_tensor.cpu().numpy(), origin='lower', 
                            extent=[-domain_size_init, domain_size_init, -domain_size_init, domain_size_init],
                            cmap='plasma')  # Using a different colormap to distinguish
    axs[1].set_title('Warped State (Adjustable Parameters)')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    fig.colorbar(right_im, ax=axs[1])
    
    # Set up titles and explanatory text
    fig.text(0.5, 0.97, 'Hyperbolic Transformation Flow via Lie Algebra', 
             ha='center', va='center', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.93, f'Left: Fixed Parameters (e={left_eccentricity}, c={left_c_factor}) | Right: Adjustable Parameters', 
             ha='center', va='center', fontsize=10)
    
    # Add rotation angle display
    rotation_text = fig.text(0.5, 0.9, 'Rotation: [0.00, 0.00, 0.00]', ha='center', va='center', color='black')
    
    # Add circles to show boundary
    circle1 = plt.Circle((0, 0), domain_size_init, fill=False, color='red', linestyle='--')
    circle2 = plt.Circle((0, 0), domain_size_init, fill=False, color='blue', linestyle='--')
    axs[0].add_patch(circle1)
    axs[1].add_patch(circle2)
    
    # Set up blend factor display
    blend_text = fig.text(0.5, 0.85, 'Flow Rate: 0.20', ha='center', va='center', color='black')
    
    # Add eccentricity and c-factor displays
    eccentricity_text = fig.text(0.25, 0.85, f'Eccentricity: {eccentricity_init:.2f}', ha='center', va='center', color='black')
    c_factor_text = fig.text(0.75, 0.85, f'C-Factor: {c_factor_init:.2f}', ha='center', va='center', color='black')
    
    # Add normalization status text
    normalize_text = fig.text(0.1, 0.85, 'Auto-Normalize: OFF', ha='center', va='center', color='black')
    
    # Adjust the layout to make room for controls
    plt.subplots_adjust(bottom=0.45)  # Increased to make room for new sliders
    
    # Add sliders for parameters
    ax_domain = plt.axes([0.25, 0.40, 0.65, 0.03])  # Domain size
    ax_radius = plt.axes([0.25, 0.35, 0.65, 0.03])  # Radius
    ax_eccentricity = plt.axes([0.25, 0.30, 0.65, 0.03])  # Eccentricity - new slider
    ax_c_factor = plt.axes([0.25, 0.25, 0.65, 0.03])  # C-Factor - new slider
    ax_theta_x = plt.axes([0.25, 0.20, 0.65, 0.03])  # X Rotation
    ax_theta_y = plt.axes([0.25, 0.15, 0.65, 0.03])  # Y Rotation
    ax_theta_z = plt.axes([0.25, 0.10, 0.65, 0.03])  # Z Rotation
    ax_anim_speed = plt.axes([0.25, 0.05, 0.65, 0.03])  # Animation Speed
    ax_blend = plt.axes([0.25, 0.00, 0.65, 0.03])  # Blending factor
    
    slider_domain = Slider(ax_domain, 'Domain Size', 0.1, 10.0, valinit=domain_size_init)
    slider_radius = Slider(ax_radius, 'Radius', 0.1, 10.0, valinit=r_init)
    # New sliders for eccentricity and c-factor
    slider_eccentricity = Slider(ax_eccentricity, 'Eccentricity', 0.1, 10.0, valinit=eccentricity_init)
    slider_c_factor = Slider(ax_c_factor, 'C-Factor', 0.1, 5.0, valinit=c_factor_init)
    slider_theta_x = Slider(ax_theta_x, 'X Rotation', -np.pi, np.pi, valinit=theta_x_init)
    slider_theta_y = Slider(ax_theta_y, 'Y Rotation', -np.pi, np.pi, valinit=theta_y_init)
    slider_theta_z = Slider(ax_theta_z, 'Z Rotation', -np.pi, np.pi, valinit=theta_z_init)
    slider_anim_speed = Slider(ax_anim_speed, 'Anim Speed', 0.1, 3.0, valinit=animation_speed_init)
    slider_blend = Slider(ax_blend, 'Flow Rate', 0.0, 1.0, valinit=0.2)
    
    # Add button to regenerate function tensor
    ax_regen = plt.axes([0.05, 0.35, 0.15, 0.05])
    button_regen = Button(ax_regen, 'Regenerate Function')
    
    # Add animation control button
    ax_anim_toggle = plt.axes([0.05, 0.25, 0.15, 0.05])
    button_anim_toggle = Button(ax_anim_toggle, 'Start Animation')
    
    # Add button to reset state to original
    ax_reset_state = plt.axes([0.05, 0.15, 0.15, 0.05])
    button_reset_state = Button(ax_reset_state, 'Reset State')
    
    # Add normalization controls
    ax_normalize_now = plt.axes([0.05, 0.05, 0.15, 0.05])
    button_normalize_now = Button(ax_normalize_now, 'Normalize Now')
    
    # Add auto-normalize toggle
    ax_auto_normalize = plt.axes([0.05, 0.00, 0.15, 0.05])
    button_auto_normalize = Button(ax_auto_normalize, 'Toggle Auto-Norm')
    
    # Function to update both panels with current and warped states
    def update_visualizations(domain_size, r, rotation_angles, eccentricity, c_factor):
        """Update visualizations with current state and warped state"""
        nonlocal right_function_tensor, right_ball_coords, right_flat_coords
        
        # Update circle radius
        circle1.set_radius(domain_size)
        circle2.set_radius(domain_size)
        
        # Scale the normalized coordinates by the domain size
        scaled_coords = normalized_coords_torch * domain_size
        
        with torch.no_grad():  # Disable gradient tracking
            # Convert the scaled coordinates to flat space using the right plot parameters
            ball_coords_for_warp = scaled_coords.clone()  # These are in ball space
            flat_coords_for_warp = ball_to_flat(ball_coords_for_warp, eccentricity, c_factor)
            
            # Apply warping to the flat space coordinates
            warped_flat_coords = warp_with_rotation(
                flat_coords_for_warp, 
                r, 
                create_nd_rotation_matrix(rotation_angles, 3).to(device)
            )
            
            # Convert warped coordinates back to ball space for sampling
            warped_ball_coords = flat_to_ball(warped_flat_coords, eccentricity, c_factor)
            
            # Sample the function at the warped ball coordinates
            warped_values = sample_function_with_interpolation(
                warped_ball_coords, right_function_tensor, right_ball_coords, resolution
            )
            
            # Reshape for visualization
            warped_data = warped_values.reshape(resolution, resolution)
            
            # Get blend factor
            blend_factor = slider_blend.val
            
            # Display the warped state in right panel
            right_im.set_array(warped_data.cpu().numpy())
            
            # At each time step, update the right function tensor by flowing along the Lie algebra direction
            # Apply normalization to prevent value drift over time
            updated_tensor = (1 - blend_factor) * right_function_tensor + blend_factor * warped_data
            
            # Apply normalization if auto_normalize is enabled
            if auto_normalize:
                updated_tensor = normalize_tensor(updated_tensor)
                
            right_function_tensor = updated_tensor
        
        # Update color scales - only for the right image
        right_im.set_clim(right_function_tensor.min().item(), right_function_tensor.max().item())
        
        # Update axis limits
        for ax in axs:
            ax.set_xlim(-domain_size, domain_size)
            ax.set_ylim(-domain_size, domain_size)
    
    # Function to regenerate the function tensor
    def regenerate_function(event=None):
        nonlocal left_function_tensor, left_ball_coords, left_flat_coords
        nonlocal right_function_tensor, right_ball_coords, right_flat_coords
        nonlocal original_left_tensor, original_left_ball_coords, original_left_flat_coords
        nonlocal original_right_tensor, original_right_ball_coords, original_right_flat_coords
        
        domain_size = slider_domain.val
        eccentricity = slider_eccentricity.val
        c_factor = slider_c_factor.val
        
        # Free memory by explicitly deleting old tensors
        del left_function_tensor, left_ball_coords, left_flat_coords
        del right_function_tensor, right_ball_coords, right_flat_coords
        del original_left_tensor, original_left_ball_coords, original_left_flat_coords
        del original_right_tensor, original_right_ball_coords, original_right_flat_coords
        
        # Create new function tensor for left plot (with fixed parameters)
        left_function_tensor, left_ball_coords, left_flat_coords = create_function_tensor(
            resolution, domain_size, left_eccentricity, left_c_factor, device
        )
        
        # Create new function tensor for right plot (with adjustable parameters)
        right_function_tensor, right_ball_coords, right_flat_coords = create_function_tensor(
            resolution, domain_size, eccentricity, c_factor, device
        )
        
        # Update original tensor references
        with torch.no_grad():  # Disable gradient tracking
            original_left_tensor = left_function_tensor.clone()
            original_left_ball_coords = left_ball_coords.clone()
            original_left_flat_coords = left_flat_coords.clone()
            
            original_right_tensor = right_function_tensor.clone()
            original_right_ball_coords = right_ball_coords.clone()
            original_right_flat_coords = right_flat_coords.clone()
        
        # Update both displays
        left_im.set_array(left_function_tensor.cpu().numpy())
        left_im.set_extent([-domain_size, domain_size, -domain_size, domain_size])
        left_im.set_clim(left_function_tensor.min().item(), left_function_tensor.max().item())
        
        # Set right display
        right_im.set_array(right_function_tensor.cpu().numpy())
        right_im.set_extent([-domain_size, domain_size, -domain_size, domain_size])
        right_im.set_clim(right_function_tensor.min().item(), right_function_tensor.max().item())
        
        # Clear the figure cache to reduce memory usage
        fig.canvas.flush_events()
        fig.canvas.draw_idle()
    
    # Function to reset the state to the original function
    def reset_state(event=None):
        nonlocal left_function_tensor, original_left_tensor
        nonlocal left_ball_coords, original_left_ball_coords
        nonlocal left_flat_coords, original_left_flat_coords
        
        nonlocal right_function_tensor, original_right_tensor
        nonlocal right_ball_coords, original_right_ball_coords
        nonlocal right_flat_coords, original_right_flat_coords
        
        with torch.no_grad():  # Disable gradient tracking
            # Reset left plot
            left_function_tensor = original_left_tensor.clone()
            left_ball_coords = original_left_ball_coords.clone()
            left_flat_coords = original_left_flat_coords.clone()
            
            # Reset right plot
            right_function_tensor = original_right_tensor.clone()
            right_ball_coords = original_right_ball_coords.clone()
            right_flat_coords = original_right_flat_coords.clone()
            
        # Update the displays
        left_im.set_array(left_function_tensor.cpu().numpy())
        left_im.set_clim(left_function_tensor.min().item(), left_function_tensor.max().item())
        
        right_im.set_array(right_function_tensor.cpu().numpy())
        right_im.set_clim(right_function_tensor.min().item(), right_function_tensor.max().item())
        
        # Clear the figure cache to reduce memory usage
        fig.canvas.flush_events()
        fig.canvas.draw_idle()
    
    # Function to perform immediate normalization
    def normalize_now(event=None):
        nonlocal right_function_tensor
        
        with torch.no_grad():
            # Normalize the right function tensor
            right_function_tensor = normalize_tensor(right_function_tensor)
            
            # Update the display
            right_im.set_array(right_function_tensor.cpu().numpy())
            right_im.set_clim(0, 1)  # Set fixed color limits for normalized data
            
        # Clear the figure cache to reduce memory usage
        fig.canvas.flush_events()
        fig.canvas.draw_idle()
    
    # Function to toggle auto-normalization
    def toggle_auto_normalize(event=None):
        nonlocal auto_normalize
        
        # Toggle the auto-normalize flag
        auto_normalize = not auto_normalize
        
        # Update the button text and status display
        if auto_normalize:
            button_auto_normalize.label.set_text('Auto-Norm: ON')
            normalize_text.set_text('Auto-Normalize: ON')
            normalize_text.set_color('green')
            # Normalize immediately when turning on
            normalize_now()
        else:
            button_auto_normalize.label.set_text('Auto-Norm: OFF')
            normalize_text.set_text('Auto-Normalize: OFF')
            normalize_text.set_color('black')
        
        fig.canvas.draw_idle()
    
    # Add a frame counter to limit updates and prevent slowdown
    frame_counter = 0
    update_frequency = 2  # Update every N frames
    
    # This function will be called for each animation frame
    def animate(frame):
        if not animation_running:
            return [right_im]
        
        # Use a frame counter to reduce update frequency and prevent slowdown
        nonlocal frame_counter
        frame_counter += 1
        if frame_counter % update_frequency != 0:
            return [right_im]
        
        # Get base parameter values from sliders
        domain_size = slider_domain.val
        r = slider_radius.val
        eccentricity = slider_eccentricity.val  # Get eccentricity from slider
        c_factor = slider_c_factor.val  # Get c_factor from slider
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
        blend_text.set_text(f'Flow Rate: {blend_factor:.2f}')
        
        # Update eccentricity and c-factor displays
        eccentricity_text.set_text(f'Eccentricity: {eccentricity:.2f}')
        c_factor_text.set_text(f'C-Factor: {c_factor:.2f}')
        
        # Update both visualizations
        update_visualizations(domain_size, r, [theta_x, theta_y, theta_z], eccentricity, c_factor)
        
        # Explicitly clear unused objects for garbage collection
        import gc
        gc.collect()
        
        return [right_im]
    
    # Initialize the animation object with save_count=1 to prevent memory growth
    anim = FuncAnimation(fig, animate, frames=1000, interval=30, blit=True, save_count=1)
    anim.pause()  # Start paused
    
    # Function to update display when sliders change
    def update(val=None):
        if not animation_running:  # Only update manually if animation is not running
            # Get current parameter values
            domain_size = slider_domain.val
            r = slider_radius.val
            eccentricity = slider_eccentricity.val  # Get eccentricity from slider
            c_factor = slider_c_factor.val  # Get c_factor from slider
            theta_x = slider_theta_x.val
            theta_y = slider_theta_y.val
            theta_z = slider_theta_z.val
            blend_factor = slider_blend.val
            
            # Update blend factor display
            blend_text.set_text(f'Flow Rate: {blend_factor:.2f}')
            
            # Update eccentricity and c-factor displays
            eccentricity_text.set_text(f'Eccentricity: {eccentricity:.2f}')
            c_factor_text.set_text(f'C-Factor: {c_factor:.2f}')
            
            # Check if domain size changed, if so regenerate function tensor
            if val is slider_domain:
                regenerate_function()
                return
                
            # Check if eccentricity or c_factor changed, if so regenerate right function tensor
            if val is slider_eccentricity or val is slider_c_factor:
                # Update the title of the right plot to reflect new parameters
                axs[1].set_title(f'Warped State (e={eccentricity:.1f}, c={c_factor:.1f})')
                
                # We only regenerate the right function tensor when these parameters change
                with torch.no_grad():
                    # Create new function tensor for right plot with updated parameters
                    nonlocal right_function_tensor, right_ball_coords, right_flat_coords
                    right_function_tensor, right_ball_coords, right_flat_coords = create_function_tensor(
                        resolution, domain_size, eccentricity, c_factor, device
                    )
                    
                    # Update original tensor references for right plot
                    nonlocal original_right_tensor, original_right_ball_coords, original_right_flat_coords
                    original_right_tensor = right_function_tensor.clone()
                    original_right_ball_coords = right_ball_coords.clone()
                    original_right_flat_coords = right_flat_coords.clone()
                    
                    # Update right display
                    right_im.set_array(right_function_tensor.cpu().numpy())
                    right_im.set_clim(right_function_tensor.min().item(), right_function_tensor.max().item())
                    
                fig.canvas.draw_idle()
                return
            
            # Reset rotation offsets when manually updating
            nonlocal theta_x_offset, theta_y_offset, theta_z_offset
            theta_x_offset = 0.0
            theta_y_offset = 0.0
            theta_z_offset = 0.0
            
            # Update the rotation text
            rotation_text.set_text(f'Rotation: [{theta_x:.2f}, {theta_y:.2f}, {theta_z:.2f}]')
            
            # Update visualizations
            update_visualizations(domain_size, r, [theta_x, theta_y, theta_z], eccentricity, c_factor)
            
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
    slider_eccentricity.on_changed(update)  # Register new slider
    slider_c_factor.on_changed(update)  # Register new slider
    slider_theta_x.on_changed(update)
    slider_theta_y.on_changed(update)
    slider_theta_z.on_changed(update)
    slider_anim_speed.on_changed(update)
    slider_blend.on_changed(update)  
    button_anim_toggle.on_clicked(toggle_animation)
    button_regen.on_clicked(regenerate_function)
    button_reset_state.on_clicked(reset_state)
    button_normalize_now.on_clicked(normalize_now)  # Register normalization button
    button_auto_normalize.on_clicked(toggle_auto_normalize)  # Register auto-normalize toggle
    
    # Initial update
    update()
    
    plt.tight_layout(rect=[0, 0.45, 1, 0.95])  # Adjusted to account for new sliders
    plt.show()

if __name__ == '__main__':
    main()
