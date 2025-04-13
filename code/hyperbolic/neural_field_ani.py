
"""
Dynamic visualization for n-dimensional stereographic projections with animation.
Uses neural fields from neural_field.py and updates the fields with gradient descent
to minimize the difference between original and warped views.
Includes noise injection, momentum, and heavy-tailed coordinate sampling.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.animation import FuncAnimation
from scipy import stats

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
    """
    device = coords.device
    
    # Create rotation matrix on the same device as coords
    rotation_matrix = create_nd_rotation_matrix(rotation_angles, 3).to(device)
    
    flat_points = ball_to_flat(coords, eccentricity, c_factor)
    warped_points = warp_with_rotation(flat_points, r, rotation_matrix)
    return warped_points

def generate_heavy_tailed_coordinates(n_points, domain_size, alpha=1.5, device='cpu'):
    """
    Generate coordinates with a heavy-tailed distribution.
    Uses a Student's t-distribution with specified alpha parameter.
    
    Parameters:
    -----------
    n_points : int
        Number of points to generate
    domain_size : float
        Domain size (coordinates will be scaled to [-domain_size, domain_size])
    alpha : float
        Parameter controlling the tail heaviness (smaller = heavier tails)
    device : torch.device
        Device to put the generated coordinates on
        
    Returns:
    --------
    torch.Tensor
        Coordinates tensor of shape (n_points, 2)
    """
    # Use the Student's t-distribution (heavy-tailed)
    df = alpha  # degrees of freedom parameter (lower = heavier tails)
    
    # Generate random points from t-distribution
    x = stats.t.rvs(df=df, size=n_points)
    y = stats.t.rvs(df=df, size=n_points)
    
    # Scale to the domain size (first normalize to roughly [-1, 1])
    max_val = max(np.abs(x).max(), np.abs(y).max())
    x = x / max_val * domain_size
    y = y / max_val * domain_size
    
    # Create tensor and move to device
    coords = np.stack([x, y], axis=1)
    return torch.tensor(coords, dtype=torch.float32, device=device)

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
    
    # Create optimizers for each field
    optimizers = []
    for field in scalar_fields:
        optimizers.append(torch.optim.SGD(field.model.parameters(), lr=0.001, momentum=0.9))
    
    # Define function dictionary with the neural fields
    function_dict = {}
    for i, field in enumerate(scalar_fields):
        function_dict[f'neural_field_{i+1}'] = field
    
    # Set up grid parameters
    resolution = 200  # Reduced for faster rendering
    sample_points = 5000  # Number of points to sample for gradient updates
    
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
    noise_level_init = 0.02  # Initial noise level
    momentum_init = 0.9  # Initial momentum value
    alpha_init = 1.5  # Initial alpha parameter for heavy-tailed distribution
    
    # Animation angle offsets - these will be used to calculate actual angles during animation
    # without updating the sliders
    theta_x_offset = 0.0
    theta_y_offset = 0.0
    theta_z_offset = 0.0
    
    # Initial function choice
    current_function_name = list(function_dict.keys())[0]
    current_function = function_dict[current_function_name]
    current_optimizer = optimizers[list(function_dict.keys()).index(current_function_name)]
    
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
    
    # Add rotation angle display
    rotation_text = fig.text(0.5, 0.9, 'Rotation: [0.00, 0.00, 0.00]', ha='center', va='center', color='black')
    
    # Add loss text display
    loss_text = fig.text(0.5, 0.95, 'Loss: N/A', ha='center', va='center', color='black')
    
    # Add circles to show boundary
    circle1 = plt.Circle((0, 0), r_init, fill=False, color='red', linestyle='--')
    circle2 = plt.Circle((0, 0), r_init, fill=False, color='red', linestyle='--')
    axs[0].add_patch(circle1)
    axs[1].add_patch(circle2)
    
    # Track sampled points (for visualization)
    sampled_points_scatter = None
    
    # Adjust the layout to make room for controls
    plt.subplots_adjust(bottom=0.5)
    
    # Add sliders for parameters
    ax_domain = plt.axes([0.25, 0.45, 0.65, 0.03])  # Domain size
    ax_radius = plt.axes([0.25, 0.40, 0.65, 0.03])  # Radius
    ax_theta_x = plt.axes([0.25, 0.35, 0.65, 0.03])  # X Rotation
    ax_theta_y = plt.axes([0.25, 0.30, 0.65, 0.03])  # Y Rotation
    ax_theta_z = plt.axes([0.25, 0.25, 0.65, 0.03])  # Z Rotation
    ax_anim_speed = plt.axes([0.25, 0.20, 0.65, 0.03])  # Animation Speed
    ax_learning_rate = plt.axes([0.25, 0.15, 0.65, 0.03])  # Learning Rate
    ax_momentum = plt.axes([0.25, 0.10, 0.65, 0.03])  # Momentum
    ax_noise = plt.axes([0.25, 0.05, 0.65, 0.03])  # Noise Level
    ax_alpha = plt.axes([0.25, 0.00, 0.65, 0.03])  # Alpha parameter for heavy-tailed distribution
    
    slider_domain = Slider(ax_domain, 'Domain Size', 0.1, 10.0, valinit=domain_size_init)
    slider_radius = Slider(ax_radius, 'Radius', 0.1, 10.0, valinit=r_init)
    slider_theta_x = Slider(ax_theta_x, 'X Rotation', -np.pi, np.pi, valinit=theta_x_init)
    slider_theta_y = Slider(ax_theta_y, 'Y Rotation', -np.pi, np.pi, valinit=theta_y_init)
    slider_theta_z = Slider(ax_theta_z, 'Z Rotation', -np.pi, np.pi, valinit=theta_z_init)
    slider_anim_speed = Slider(ax_anim_speed, 'Anim Speed', 0.1, 3.0, valinit=animation_speed_init)
    slider_learning_rate = Slider(ax_learning_rate, 'Learning Rate', 0.0001, 0.01, valinit=learning_rate_init)
    slider_momentum = Slider(ax_momentum, 'Momentum', 0.0, 0.99, valinit=momentum_init)
    slider_noise = Slider(ax_noise, 'Noise Level', 0.0, 0.1, valinit=noise_level_init)
    slider_alpha = Slider(ax_alpha, 'Tail Weight (α)', 0.5, 5.0, valinit=alpha_init)
    
    # Add animation control buttons
    ax_anim_toggle = plt.axes([0.05, 0.30, 0.15, 0.05])
    button_anim_toggle = Button(ax_anim_toggle, 'Start Animation')
    
    ax_grad_toggle = plt.axes([0.05, 0.20, 0.15, 0.05])
    button_grad_toggle = Button(ax_grad_toggle, 'Enable Gradients')
    
    ax_reset = plt.axes([0.05, 0.10, 0.15, 0.05])
    button_reset = Button(ax_reset, 'Reset Field')
    
    ax_show_samples = plt.axes([0.05, 0.40, 0.15, 0.05])
    button_show_samples = Button(ax_show_samples, 'Show Samples')
    show_samples = False
    
    # Create radio buttons for function selection
    ax_func = plt.axes([0.05, 0.00, 0.15, 0.10])
    radio = RadioButtons(ax_func, list(function_dict.keys()), active=list(function_dict.keys()).index(current_function_name))
    
    # Function to evaluate neural field with noise
    def evaluate_with_noise(field, coords, noise_level):
        """
        Evaluate the neural field at given coordinates with added noise.
        """
        # Evaluate the field
        with torch.no_grad():
            field.model.eval()
            values = field.model(coords)
            
            # Add Gaussian noise during training to help escape local minima
            if noise_level > 0:
                noise = torch.randn_like(values) * noise_level
                values = values + noise
                
                # Ensure values stay in [0, 1] range (for visualization)
                values = torch.clamp(values, 0, 1)
                
            # Convert to numpy for visualization
            if isinstance(values, torch.Tensor):
                values = values.cpu().numpy()
                
            return values
    
    # Function to perform gradient updates on the neural field
    def perform_gradient_update(field, optimizer, domain_size, r, rotation_angles, 
                               learning_rate, noise_level, momentum, alpha):
        """
        Update the neural field using sampled coordinates with heavy-tailed distribution.
        
        Returns the loss value and the sampled coordinates.
        """
        # Set learning rate and momentum
        if isinstance(optimizer, torch.optim.SGD):
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
                param_group['momentum'] = momentum
        else:
            # For other optimizers (like Adam), just set learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        
        # Sample coordinates using heavy-tailed distribution
        original_coords = generate_heavy_tailed_coordinates(
            sample_points, 
            domain_size, 
            alpha=alpha, 
            device=device
        )
        
        # Apply warping to get corresponding warped coordinates
        warped_coords = apply_warp_to_coords(
            original_coords, r, rotation_angles, eccentricity_init, c_factor_init
        )
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Put models in training mode
        field.model.train()
        
        # Forward pass for original coordinates
        original_values = field.model(original_coords)
        
        # Add noise to values to help escape local minima
        if noise_level > 0:
            original_noise = torch.randn_like(original_values) * noise_level
            original_values = original_values + original_noise
        
        # Forward pass for warped coordinates
        warped_values = field.model(warped_coords)
        
        # Add noise to values to help escape local minima
        if noise_level > 0:
            warped_noise = torch.randn_like(warped_values) * noise_level
            warped_values = warped_values + warped_noise
        
        # Compute loss (MSE between original and warped evaluations)
        loss = torch.nn.functional.mse_loss(original_values, warped_values)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Return the loss value and the sampled points
        return loss.item(), original_coords.cpu().numpy()
    
    # Function to update both original and warped visualizations
    def update_visualizations(domain_size, r, rotation_angles, noise_level):
        """Update both original and warped visualizations with current parameters"""
        # Update circle radius
        circle1.set_radius(r)
        circle2.set_radius(r)
        
        # Scale the normalized grid by the domain size for original view
        X = normalized_X * domain_size
        Y = normalized_Y * domain_size
        
        # Scale the normalized coordinates by the domain size
        scaled_coords = normalized_coords_torch * domain_size
        
        # Apply warping to the coordinates
        warped_coords = apply_warp_to_coords(
            scaled_coords, r, rotation_angles, eccentricity_init, c_factor_init
        )
        
        # Reshape coordinates for the neural field
        warped_x = warped_coords[:, 0].reshape(resolution, resolution).cpu().numpy()
        warped_y = warped_coords[:, 1].reshape(resolution, resolution).cpu().numpy()
        
        # Update the original image - evaluate with noise
        original_values = evaluate_with_noise(current_function, scaled_coords, noise_level)
        original_data = original_values.reshape(resolution, resolution)
        original_im.set_array(original_data)
        original_im.set_extent([-domain_size, domain_size, -domain_size, domain_size])
        original_im.set_clim(original_data.min(), original_data.max())
        
        # Update the warped image - evaluate with the same noise level
        warped_values = evaluate_with_noise(current_function, warped_coords, noise_level)
        warped_data = warped_values.reshape(resolution, resolution)
        warped_im.set_array(warped_data)
        warped_im.set_extent([-domain_size, domain_size, -domain_size, domain_size])
        warped_im.set_clim(warped_data.min(), warped_data.max())
        
        # Update axis limits
        for ax in axs:
            ax.set_xlim(-domain_size, domain_size)
            ax.set_ylim(-domain_size, domain_size)
        
        return scaled_coords, warped_coords
    
    # Function to toggle sample point visualization
    def toggle_samples(event):
        nonlocal show_samples, sampled_points_scatter
        show_samples = not show_samples
        
        if show_samples:
            button_show_samples.label.set_text('Hide Samples')
            
            # Generate and display sample points
            alpha = slider_alpha.val
            domain_size = slider_domain.val
            sampled_coords = generate_heavy_tailed_coordinates(
                sample_points, 
                domain_size, 
                alpha=alpha, 
                device='cpu'
            ).numpy()
            
            # Remove existing scatter if it exists
            if sampled_points_scatter is not None:
                for scatter in sampled_points_scatter:
                    scatter.remove()
            
            # Plot samples on both axes
            sampled_points_scatter = []
            for ax in axs:
                scatter = ax.scatter(
                    sampled_coords[:, 0], 
                    sampled_coords[:, 1], 
                    s=2, 
                    color='red', 
                    alpha=0.5
                )
                sampled_points_scatter.append(scatter)
        else:
            button_show_samples.label.set_text('Show Samples')
            
            # Remove scatter plots
            if sampled_points_scatter is not None:
                for scatter in sampled_points_scatter:
                    scatter.remove()
                sampled_points_scatter = None
        
        fig.canvas.draw_idle()
    
    # This function will be called for each animation frame
    def animate(frame):
        if not animation_running:
            return [warped_im]
        
        # Get base parameter values from sliders
        domain_size = slider_domain.val
        r = slider_radius.val
        base_theta_x = slider_theta_x.val
        base_theta_y = slider_theta_y.val
        base_theta_z = slider_theta_z.val
        learning_rate = slider_learning_rate.val
        noise_level = slider_noise.val
        momentum = slider_momentum.val
        alpha = slider_alpha.val
        speed = slider_anim_speed.val
        
        # Calculate rotation angles based on frame and animation speed
        # Now we use offsets instead of updating the sliders
        nonlocal theta_x_offset, theta_y_offset, theta_z_offset
        theta_x_offset = 0.05 * speed * np.sin(0.01 * frame)
        theta_y_offset = 0.05 * speed * np.sin(0.02 * frame)
        theta_z_offset = 0.05 * speed * np.cos(0.015 * frame)
        
        # Calculate actual angles for this frame
        theta_x = base_theta_x + theta_x_offset
        theta_y = base_theta_y + theta_y_offset
        theta_z = base_theta_z + theta_z_offset
        
        # Update the rotation text display instead of the sliders
        rotation_text.set_text(f'Rotation: [{theta_x:.2f}, {theta_y:.2f}, {theta_z:.2f}]')
        
        # Update both visualizations and get the coordinates
        # Use a lower noise level for visualization to avoid flickering
        vis_noise_level = 0 if not gradient_updates_enabled else noise_level * 0.3
        update_visualizations(domain_size, r, [theta_x, theta_y, theta_z], vis_noise_level)
        
        # Perform gradient update if enabled
        if gradient_updates_enabled:
            # Perform gradient updates, potentially multiple steps per frame
            steps_per_frame = 1
            loss_val = 0
            sampled_coords = None
            
            for _ in range(steps_per_frame):
                loss_val, sampled_coords = perform_gradient_update(
                    current_function, 
                    current_optimizer, 
                    domain_size,
                    r, 
                    [theta_x, theta_y, theta_z], 
                    learning_rate,
                    noise_level,
                    momentum,
                    alpha
                )
            
            # Update the visualizations again after the gradient update
            update_visualizations(domain_size, r, [theta_x, theta_y, theta_z], vis_noise_level)
            
            # Update the loss text
            loss_text.set_text(f'Loss: {loss_val:.6f}, α: {alpha:.2f}, Samples: {sample_points}')
            
            # Update sample point visualization if enabled
            if show_samples and sampled_coords is not None:
                nonlocal sampled_points_scatter
                
                # Remove existing scatter if it exists
                if sampled_points_scatter is not None:
                    for scatter in sampled_points_scatter:
                        scatter.remove()
                
                # Plot samples on both axes
                sampled_points_scatter = []
                for ax in axs:
                    scatter = ax.scatter(
                        sampled_coords[:, 0], 
                        sampled_coords[:, 1], 
                        s=2, 
                        color='red', 
                        alpha=0.5
                    )
                    sampled_points_scatter.append(scatter)
        
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
            noise_level = slider_noise.val * 0.3  # Lower noise for manual updates
            
            # Reset rotation offsets when manually updating
            nonlocal theta_x_offset, theta_y_offset, theta_z_offset
            theta_x_offset = 0.0
            theta_y_offset = 0.0
            theta_z_offset = 0.0
            
            # Update the rotation text
            rotation_text.set_text(f'Rotation: [{theta_x:.2f}, {theta_y:.2f}, {theta_z:.2f}]')
            
            # Update both visualizations
            update_visualizations(domain_size, r, [theta_x, theta_y, theta_z], noise_level)
            
            # Update sample point visualization if enabled
            if show_samples:
                alpha = slider_alpha.val
                sampled_coords = generate_heavy_tailed_coordinates(
                    sample_points, 
                    domain_size, 
                    alpha=alpha, 
                    device='cpu'
                ).numpy()
                
                nonlocal sampled_points_scatter
                
                # Remove existing scatter if it exists
                if sampled_points_scatter is not None:
                    for scatter in sampled_points_scatter:
                        scatter.remove()
                
                # Plot samples on both axes
                sampled_points_scatter = []
                for ax in axs:
                    scatter = ax.scatter(
                        sampled_coords[:, 0], 
                        sampled_coords[:, 1], 
                        s=2, 
                        color='red', 
                        alpha=0.5
                    )
                    sampled_points_scatter.append(scatter)
            
            fig.canvas.draw_idle()
    
    # Function to update optimizer parameter when sliders change
    def update_optimizer_params(val=None):
        momentum = slider_momentum.val
        learning_rate = slider_learning_rate.val
        
        # Update the optimizer parameters
        if isinstance(current_optimizer, torch.optim.SGD):
            for param_group in current_optimizer.param_groups:
                param_group['momentum'] = momentum
                param_group['lr'] = learning_rate
        else:
            # For other optimizers (like Adam), just set learning rate
            for param_group in current_optimizer.param_groups:
                param_group['lr'] = learning_rate
    
    # Function to update the neural field selection
    def update_function(function_name):
        nonlocal current_function, current_function_name, current_optimizer
        current_function_name = function_name
        current_function = function_dict[function_name]
        current_optimizer = optimizers[list(function_dict.keys()).index(function_name)]
        
        # Update optimizer parameters with current slider values
        update_optimizer_params()
        
        # Force a full update to refresh everything
        update()
    
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
    
    # Function to toggle gradient updates
    def toggle_gradients(event):
        nonlocal gradient_updates_enabled
        gradient_updates_enabled = not gradient_updates_enabled
        
        # Update optimizer parameters with current slider values
        update_optimizer_params()
        
        if gradient_updates_enabled:
            button_grad_toggle.label.set_text('Disable Gradients')
            alpha = slider_alpha.val
            loss_text.set_text(f'Loss: 0.000000, α: {alpha:.2f}, Samples: {sample_points}')
        else:
            button_grad_toggle.label.set_text('Enable Gradients')
            loss_text.set_text('Loss: N/A')
        
        fig.canvas.draw_idle()
    
    # Function to reset the neural field
    def reset_field(event):
        # Recreate the current field with the same parameters
        idx = list(function_dict.keys()).index(current_function_name)
        new_field = create_random_fields(num_fields=1, seed=42+idx, output_dim=1, device=device)[0]
        
        # Replace in the dictionary
        function_dict[current_function_name] = new_field
        
        # Update current function reference
        nonlocal current_function
        current_function = new_field
        
        # Create a new optimizer with current momentum
        momentum = slider_momentum.val
        learning_rate = slider_learning_rate.val
        nonlocal current_optimizer
        current_optimizer = torch.optim.SGD(new_field.model.parameters(), 
                                           lr=learning_rate, 
                                           momentum=momentum)
        optimizers[idx] = current_optimizer
        
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
    slider_noise.on_changed(update)
    slider_alpha.on_changed(update)
    slider_learning_rate.on_changed(update_optimizer_params)
    slider_momentum.on_changed(update_optimizer_params)
    radio.on_clicked(update_function)
    button_anim_toggle.on_clicked(toggle_animation)
    button_grad_toggle.on_clicked(toggle_gradients)
    button_reset.on_clicked(reset_field)
    button_show_samples.on_clicked(toggle_samples)
    
    # Initial update
    update()
    
    plt.tight_layout(rect=[0, 0.5, 1, 0.95])  # Adjust for the loss text at the top
    plt.show()

if __name__ == '__main__':
    main()
