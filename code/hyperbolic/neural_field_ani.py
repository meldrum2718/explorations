
"""
Dynamic visualization for n-dimensional stereographic projections with animation.
Uses neural fields from neural_field.py and updates the fields with gradient descent
to minimize the difference between original and warped views.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
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
from neural_field import NeuralField


def generate_coords(xmin=-1, xmax=1, ymin=-1, ymax=1, resolution=100, device='cpu'):
    x = torch.linspace(xmin, xmax, resolution, device=device)
    y = torch.linspace(ymin, ymax, resolution, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    coords = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    return coords


def main():
    # Set device for torch
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 
                          'cpu')
    print(f"Using device: {device}")
    
    seed = 42

    # Create random neural fields (scalar only)
    d_h = 64
    L = 10
    n_layers = 3
    field = NeuralField(seed, d_h=d_h, n_layers=n_layers, L=L, output_dim=1, device=device)

    optimizer = torch.optim.SGD(field.model.parameters(), lr=0.11, momentum=0.9)
    
    resolution = 200
    sample_points = 5000
    
    domain_size = 2.0
    r = 1.0
    eccentricity = 3.0
    c_factor = 1.30
    
    theta_x = 0.0
    theta_y = 0.0
    theta_z = 0.0
    rotation_angles = [theta_x, theta_y, theta_z]
    rotation_matrix = create_nd_rotation_matrix(rotation_angles, dim=3).to(device)

    
    gradient_updates_enabled = False  # Flag to enable/disable gradient updates
    learning_rate = 0.001  # Initial learning rate
    momentum = 0.9  # Initial momentum value
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    flat_coords = generate_coords(xmin=-domain_size, xmax=domain_size, ymin=-domain_size, ymax=domain_size, resolution=resolution, device=device)

    warped_coords = warp_with_rotation(flat_coords, r, rotation_matrix)
    flat_state = field.model(flat_coords)
    warped_state = field.model(warped_coords)

    flat_im = axs[0].imshow(flat_state.cpu().detach().reshape(resolution, resolution), cmap='viridis')
    axs[0].set_title('Current Neural Field')
    fig.colorbar(flat_im, ax=axs[0])
    
    warped_im = axs[1].imshow(warped_state.cpu().detach().reshape(resolution, resolution), cmap='viridis')
    axs[1].set_title('Warped Neural Field')
    fig.colorbar(warped_im, ax=axs[1])
    
    # Adjust the layout to make room for controls
    plt.subplots_adjust(bottom=0.5)
    
    # Add sliders for parameters
    ax_domain = plt.axes([0.25, 0.45, 0.65, 0.03])  # Domain size
    ax_radius = plt.axes([0.25, 0.40, 0.65, 0.03])  # Radius
    ax_theta_x = plt.axes([0.25, 0.35, 0.65, 0.03])  # X Rotation
    ax_theta_y = plt.axes([0.25, 0.30, 0.65, 0.03])  # Y Rotation
    ax_theta_z = plt.axes([0.25, 0.25, 0.65, 0.03])  # Z Rotation
    ax_learning_rate = plt.axes([0.25, 0.15, 0.65, 0.03])  # Learning Rate
    ax_momentum = plt.axes([0.25, 0.10, 0.65, 0.03])  # Momentum
    
    slider_domain = Slider(ax_domain, 'Domain Size', valmin=0.1, valmax=1000.0, valinit=domain_size)
    slider_radius = Slider(ax_radius, 'Radius', valmin=0.1, valmax=10.0, valinit=r)
    slider_theta_x = Slider(ax_theta_x, 'X Rotation', valmin=-np.pi, valmax=np.pi, valinit=theta_x)
    slider_theta_y = Slider(ax_theta_y, 'Y Rotation', valmin=-np.pi, valmax=np.pi, valinit=theta_y)
    slider_theta_z = Slider(ax_theta_z, 'Z Rotation', valmin=-np.pi, valmax=np.pi, valinit=theta_z)
    slider_learning_rate = Slider(ax_learning_rate, 'Learning Rate', valmin=0.0001, valmax=0.31, valinit=learning_rate)
    slider_momentum = Slider(ax_momentum, 'Momentum', valmin=0.0, valmax=0.99, valinit=momentum)
    
    ax_grad_toggle = plt.axes([0.05, 0.20, 0.15, 0.05])
    button_grad_toggle = Button(ax_grad_toggle, 'Enable Gradients')
    
    ax_reset = plt.axes([0.05, 0.10, 0.15, 0.05])
    button_reset = Button(ax_reset, 'Reset Field')
    
    # Function to perform gradient updates on the neural field
    def perform_gradient_update(field, optimizer, coords, target_values, learning_rate, momentum):
        """
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
        
        optimizer.zero_grad() # Zero the gradients
        
        field.model.train()
        
        field_values = field.model(coords)
        
        loss = torch.nn.functional.mse_loss(field_values, target_values)
        
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def update_visualizations(domain_size, resolution, r, rotation_angles):
        """Update both original and warped visualizations with current parameters"""

        flat_coords = generate_coords(xmin=-domain_size, xmax=domain_size, ymin=-domain_size, ymax=domain_size, resolution=resolution, device=device)
        rotation_matrix = create_nd_rotation_matrix(rotation_angles, 3).to(device)

        warped_coords = warp_with_rotation(flat_coords, r, rotation_matrix)

        with torch.no_grad():
            field.model.eval()
            flat_state = field.model(flat_coords)
            warped_state = field.model(warped_coords)

        flat_state = torch.exp(flat_state)
        warped_state = torch.exp(warped_state)
        flat_im.set_data(flat_state.detach().cpu().reshape(resolution, resolution))
        flat_im.set_clim(flat_state.min(), flat_state.max())
        
        warped_im.set_data(warped_state.detach().cpu().reshape(resolution, resolution))
        warped_im.set_clim(warped_state.min(), warped_state.max())
        
        return flat_coords, flat_state, warped_coords, warped_state
    
    # This function will be called for each animation frame
    def animate(frame):
        
        # Get base parameter values from sliders
        domain_size = slider_domain.val
        r = slider_radius.val
        learning_rate = slider_learning_rate.val
        momentum = slider_momentum.val

        flat_coords, flat_state, warped_coords, warped_state = update_visualizations(domain_size, resolution, r, [theta_x, theta_y, theta_z])
        
        # Perform gradient update if enabled
        if gradient_updates_enabled:
            steps_per_frame = 1
            loss_val = 0
            
            for _ in range(steps_per_frame):
                loss_val += perform_gradient_update(
                    field,
                    optimizer,
                    coords=flat_coords,
                    target_values=warped_state,
                    learning_rate=learning_rate,
                    momentum=momentum
                )

            print('loss val:', loss_val)


            
        return [flat_im, warped_im]

    # Initialize the animation object
    anim = FuncAnimation(fig, animate, interval=30, blit=True, save_count=1)
    anim.pause()
    
    # Function to update display when sliders change
    def update_slider_values(val=None):
        nonlocal domain_size, r, theta_x, theta_y, theta_z
        domain_size = slider_domain.val
        r = slider_radius.val
        theta_x = slider_theta_x.val
        theta_y = slider_theta_y.val
        theta_z = slider_theta_z.val
            
        # update_visualizations(domain_size, resolution, r, [theta_x, theta_y, theta_z])
        # fig.canvas.draw_idle()
    
    ## # Function to update optimizer parameter when sliders change
    ## def update_optimizer_params(val=None):
    ##     momentum = slider_momentum.val
    ##     learning_rate = slider_learning_rate.val
    ##     
    ##     # Update the optimizer parameters
    ##     if isinstance(optimizer, torch.optim.SGD):
    ##         for param_group in optimizer.param_groups:
    ##             param_group['momentum'] = momentum
    ##             param_group['lr'] = learning_rate
    ##     else:
    ##         # For other optimizers (like Adam), just set learning rate
    ##         for param_group in optimizer.param_groups:
    ##             param_group['lr'] = learning_rate
    
    def toggle_gradients(event):
        nonlocal gradient_updates_enabled
        gradient_updates_enabled = not gradient_updates_enabled
        
        if gradient_updates_enabled:
            button_grad_toggle.label.set_text('Disable Gradients')
        else:
            button_grad_toggle.label.set_text('Enable Gradients')
        
        fig.canvas.draw_idle()
    
    # Function to reset the neural field
    def reset_field(event):
        nonlocal field, optimizer
        field = NeuralField(seed, d_h=d_h, n_layers=n_layers, L=L, output_dim=1, device=device)
        
        # Create a new optimizer with current momentum
        momentum = slider_momentum.val
        learning_rate = slider_learning_rate.val
        optimizer = torch.optim.SGD(field.model.parameters(), 
                                           lr=learning_rate, 
                                           momentum=momentum)
        # Force a full update
        # update_visualizations()
        
        # Update the loss text
        loss_text.set_text('Loss: N/A')
        
        fig.canvas.draw_idle()
    
    # Register event handlers
    slider_domain.on_changed(update_slider_values)
    slider_radius.on_changed(update_slider_values)
    slider_theta_x.on_changed(update_slider_values)
    slider_theta_y.on_changed(update_slider_values)
    slider_theta_z.on_changed(update_slider_values)
    slider_learning_rate.on_changed(update_slider_values)
    slider_momentum.on_changed(update_slider_values)
    button_grad_toggle.on_clicked(toggle_gradients)
    button_reset.on_clicked(reset_field)
    
    plt.tight_layout(rect=[0, 0.5, 1, 0.95])  # Adjust for the loss text at the top
    plt.show()

if __name__ == '__main__':
    main()
