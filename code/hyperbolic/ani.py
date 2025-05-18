import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
import torch

# Import the projection functions and FunctionTensor class
from projections import (
    ball_to_flat,
    flat_to_ball,
    warp_with_rotation,
    create_nd_rotation_matrix,
    normalize_tensor,
    FunctionTensor
)


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
    
    domain_size_init = 1.0  # Initial domain size as unit ball
    r_init = 2.0
    
    # Parameters for the tensor
    eccentricity_init = 2.0
    c_factor_init = 1.34
    
    # Create function tensor for visualization with its parameters
    # Uses a single FunctionTensor to maintain state
    function_tensor = FunctionTensor(
        resolution=resolution,
        domain_size=domain_size_init,  # Using unit ball (1.0)
        eccentricity=eccentricity_init,
        c_factor=c_factor_init,
        device=device
    )
    
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
    
    # Create a figure with two subplots (current state and warped state)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left plot - displays current state
    current_im = axs[0].imshow(function_tensor.function_tensor.cpu().numpy(), origin='lower', 
                          extent=[-domain_size_init, domain_size_init, -domain_size_init, domain_size_init],
                          cmap='viridis')
    axs[0].set_title(f'Current State (e={eccentricity_init}, c={c_factor_init})')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    fig.colorbar(current_im, ax=axs[0])
    
    # Right plot - shows warped state
    # Calculate the warped state for initial visualization
    with torch.no_grad():
        scaled_coords = normalized_coords_torch * domain_size_init
        warped_coords = function_tensor.warp_coordinates(
            scaled_coords, 
            r=r_init, 
            rotation_angles=[theta_x_init, theta_y_init, theta_z_init]
        )
        
        warped_values = function_tensor.sample_at_ball_coords(
            flat_to_ball(warped_coords, eccentricity=eccentricity_init, c_factor=c_factor_init)
        )
        warped_data = warped_values.reshape(resolution, resolution)
    
    warped_im = axs[1].imshow(warped_data.cpu().numpy(), origin='lower', 
                          extent=[-domain_size_init, domain_size_init, -domain_size_init, domain_size_init],
                          cmap='plasma')  # Using a different colormap to distinguish
    axs[1].set_title('Warped State')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    fig.colorbar(warped_im, ax=axs[1])
    
    # Set up titles and explanatory text
    fig.text(0.5, 0.97, 'Hyperbolic Transformation Flow via Lie Algebra', 
             ha='center', va='center', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.93, f'Left: Current State | Right: Warped State (e={eccentricity_init}, c={c_factor_init})', 
             ha='center', va='center', fontsize=10)
    
    # Add rotation angle display
    rotation_text = fig.text(0.5, 0.9, 'Rotation: [0.00, 0.00, 0.00]', ha='center', va='center', color='black')
    
    # Set up flow rate display
    flow_text = fig.text(0.5, 0.85, 'Flow Rate: 0.20', ha='center', va='center', color='black')
    
    # Add eccentricity and c-factor displays
    eccentricity_text = fig.text(0.25, 0.85, f'Eccentricity: {eccentricity_init:.2f}', ha='center', va='center', color='black')
    c_factor_text = fig.text(0.75, 0.85, f'C-Factor: {c_factor_init:.2f}', ha='center', va='center', color='black')
    
    # Adjust the layout to make room for controls
    plt.subplots_adjust(bottom=0.45)  # Increased to make room for sliders
    
    # Add sliders for parameters
    ax_domain = plt.axes([0.25, 0.40, 0.65, 0.03])  # Domain size
    ax_radius = plt.axes([0.25, 0.35, 0.65, 0.03])  # Radius
    ax_eccentricity = plt.axes([0.25, 0.30, 0.65, 0.03])  # Eccentricity
    ax_c_factor = plt.axes([0.25, 0.25, 0.65, 0.03])  # C-Factor
    ax_theta_x = plt.axes([0.25, 0.20, 0.65, 0.03])  # X Rotation
    ax_theta_y = plt.axes([0.25, 0.15, 0.65, 0.03])  # Y Rotation
    ax_theta_z = plt.axes([0.25, 0.10, 0.65, 0.03])  # Z Rotation
    ax_anim_speed = plt.axes([0.25, 0.05, 0.65, 0.03])  # Animation Speed
    ax_flow = plt.axes([0.25, 0.00, 0.65, 0.03])  # Flow rate factor
    
    slider_domain = Slider(ax_domain, 'Domain Size', 0.1, 10.0, valinit=domain_size_init)
    slider_radius = Slider(ax_radius, 'Radius', 0.1, 10.0, valinit=r_init)
    slider_eccentricity = Slider(ax_eccentricity, 'Eccentricity', -10.0, 10.0, valinit=eccentricity_init)
    slider_c_factor = Slider(ax_c_factor, 'C-Factor', -5.0, 5.0, valinit=c_factor_init)
    slider_theta_x = Slider(ax_theta_x, 'X Rotation', -np.pi, np.pi, valinit=theta_x_init)
    slider_theta_y = Slider(ax_theta_y, 'Y Rotation', -np.pi, np.pi, valinit=theta_y_init)
    slider_theta_z = Slider(ax_theta_z, 'Z Rotation', -np.pi, np.pi, valinit=theta_z_init)
    slider_anim_speed = Slider(ax_anim_speed, 'Anim Speed', 0.1, 3.0, valinit=animation_speed_init)
    slider_flow = Slider(ax_flow, 'Flow Rate', 0.0, 1.0, valinit=0.2)
    
    # Add button to regenerate function tensor
    ax_regen = plt.axes([0.05, 0.35, 0.15, 0.05])
    button_regen = Button(ax_regen, 'Regenerate Function')
    
    # Add animation control button
    ax_anim_toggle = plt.axes([0.05, 0.25, 0.15, 0.05])
    button_anim_toggle = Button(ax_anim_toggle, 'Start Animation')
    
    # Add button to reset state to original
    ax_reset_state = plt.axes([0.05, 0.15, 0.15, 0.05])
    button_reset_state = Button(ax_reset_state, 'Reset State')
    
    # Add normalization control
    ax_normalize_now = plt.axes([0.05, 0.05, 0.15, 0.05])
    button_normalize_now = Button(ax_normalize_now, 'Normalize Now')
    
    # Function to update both panels with current and warped states
    def update_visualizations(domain_size, r, rotation_angles, eccentricity, c_factor):
        """Update visualizations with current state and warped state"""
        # Scale the normalized coordinates by the domain size
        scaled_coords = normalized_coords_torch * domain_size
        
        with torch.no_grad():  # Disable gradient tracking
            # Warp the coordinates using the FunctionTensor's method
            warped_coords = function_tensor.warp_coordinates(
                scaled_coords, 
                r=r, 
                rotation_angles=rotation_angles
            )
            
            # Sample the function at the warped coordinates
            warped_values = function_tensor.sample_at_ball_coords(
                flat_to_ball(warped_coords, eccentricity, c_factor)
            )
            
            # Reshape for visualization
            warped_data = warped_values.reshape(resolution, resolution)
            
            # Get flow rate factor
            flow_factor = slider_flow.val
            
            # At each step, update the function tensor by flowing along the Lie algebra direction
            updated_tensor = (1 - flow_factor) * function_tensor.function_tensor + flow_factor * warped_data
            
            # Always normalize to prevent value drift over time
            updated_tensor = normalize_tensor(updated_tensor)
                
            function_tensor.set_values_from_tensor(updated_tensor)
            
            # Update current state visualization
            current_im.set_array(function_tensor.function_tensor.cpu().numpy())
            
            # Update warped state visualization
            warped_im.set_array(warped_data.cpu().numpy())
        
        # Update color scales
        current_im.set_clim(function_tensor.function_tensor.min().item(), 
                           function_tensor.function_tensor.max().item())
        warped_im.set_clim(warped_data.min().item(), warped_data.max().item())
        
        # Update axis limits
        for ax in axs:
            ax.set_xlim(-domain_size, domain_size)
            ax.set_ylim(-domain_size, domain_size)
    
    # Function to regenerate the function tensor
    def regenerate_function(event=None):
        domain_size = slider_domain.val
        eccentricity = slider_eccentricity.val
        c_factor = slider_c_factor.val
        
        # Create new function tensor with updated parameters
        function_tensor.domain_size = domain_size
        function_tensor.eccentricity = eccentricity
        function_tensor.c_factor = c_factor
        
        # Recreate the grid and compute function values
        function_tensor._create_grid()
        function_tensor._compute_function_values()
        
        # Update current state visualization
        current_im.set_array(function_tensor.function_tensor.cpu().numpy())
        current_im.set_extent([-domain_size, domain_size, -domain_size, domain_size])
        current_im.set_clim(function_tensor.function_tensor.min().item(), 
                           function_tensor.function_tensor.max().item())
        
        # Update warped state visualization
        with torch.no_grad():
            scaled_coords = normalized_coords_torch * domain_size
            warped_coords = function_tensor.warp_coordinates(
                scaled_coords, 
                r=slider_radius.val,
                rotation_angles=[slider_theta_x.val, slider_theta_y.val, slider_theta_z.val]
            )
            warped_values = function_tensor.sample_at_ball_coords(
                flat_to_ball(warped_coords, eccentricity=eccentricity, c_factor=c_factor)
            )
            warped_data = warped_values.reshape(resolution, resolution)
            
        warped_im.set_array(warped_data.cpu().numpy())
        warped_im.set_extent([-domain_size, domain_size, -domain_size, domain_size])
        warped_im.set_clim(warped_data.min().item(), warped_data.max().item())
        
        # Clear the figure cache to reduce memory usage
        fig.canvas.flush_events()
        fig.canvas.draw_idle()
    
    # Function to reset the state
    def reset_state(event=None):
        # Regenerate the function tensor instead of using stored originals
        regenerate_function()
    
    # Function to perform immediate normalization
    def normalize_now(event=None):
        with torch.no_grad():
            # Normalize the function tensor
            normalized_tensor = normalize_tensor(function_tensor.function_tensor)
            function_tensor.set_values_from_tensor(normalized_tensor)
            
            # Update the current state display
            current_im.set_array(function_tensor.function_tensor.cpu().numpy())
            current_im.set_clim(0, 1)  # Set fixed color limits for normalized data
            
            # Update warped state visualization
            domain_size = slider_domain.val
            eccentricity = slider_eccentricity.val
            c_factor = slider_c_factor.val
            r = slider_radius.val
            theta_x = slider_theta_x.val
            theta_y = slider_theta_y.val
            theta_z = slider_theta_z.val
            
            scaled_coords = normalized_coords_torch * domain_size
            warped_coords = function_tensor.warp_coordinates(
                scaled_coords, 
                r=r,
                rotation_angles=[theta_x, theta_y, theta_z]
            )
            warped_values = function_tensor.sample_at_ball_coords(
                flat_to_ball(warped_coords, eccentricity=eccentricity, c_factor=c_factor)
            )
            warped_data = warped_values.reshape(resolution, resolution)
            
            warped_im.set_array(warped_data.cpu().numpy())
            warped_im.set_clim(0, 1)
            
        # Clear the figure cache to reduce memory usage
        fig.canvas.flush_events()
        fig.canvas.draw_idle()
    
    # Add a frame counter to limit updates and prevent slowdown
    frame_counter = 0
    update_frequency = 2  # Update every N frames
    
    # This function will be called for each animation frame
    def animate(frame):
        if not animation_running:
            return [current_im, warped_im]
        
        # Use a frame counter to reduce update frequency and prevent slowdown
        nonlocal frame_counter
        frame_counter += 1
        if frame_counter % update_frequency != 0:
            return [current_im, warped_im]
        
        # Get base parameter values from sliders
        domain_size = slider_domain.val
        r = slider_radius.val
        eccentricity = slider_eccentricity.val
        c_factor = slider_c_factor.val
        base_theta_x = slider_theta_x.val
        base_theta_y = slider_theta_y.val
        base_theta_z = slider_theta_z.val
        speed = slider_anim_speed.val
        flow_factor = slider_flow.val
        
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
        flow_text.set_text(f'Flow Rate: {flow_factor:.2f}')
        
        # Update eccentricity and c-factor displays
        eccentricity_text.set_text(f'Eccentricity: {eccentricity:.2f}')
        c_factor_text.set_text(f'C-Factor: {c_factor:.2f}')
        
        # Update both visualizations
        update_visualizations(domain_size, r, [theta_x, theta_y, theta_z], eccentricity, c_factor)
        
        # Explicitly clear unused objects for garbage collection
        import gc
        gc.collect()
        
        return [current_im, warped_im]
    
    # Initialize the animation object with save_count=1 to prevent memory growth
    anim = FuncAnimation(fig, animate, interval=30, blit=True, save_count=1)
    anim.pause()  # Start paused
    
    # Function to update display when sliders change
    def update(val=None):
        if not animation_running:  # Only update manually if animation is not running
            # Get current parameter values
            domain_size = slider_domain.val
            r = slider_radius.val
            eccentricity = slider_eccentricity.val
            c_factor = slider_c_factor.val
            theta_x = slider_theta_x.val
            theta_y = slider_theta_y.val
            theta_z = slider_theta_z.val
            flow_factor = slider_flow.val
            
            # Update flow factor display
            flow_text.set_text(f'Flow Rate: {flow_factor:.2f}')
            
            # Update eccentricity and c-factor displays
            eccentricity_text.set_text(f'Eccentricity: {eccentricity:.2f}')
            c_factor_text.set_text(f'C-Factor: {c_factor:.2f}')
            
            # Check if domain size changed, if so regenerate function tensor
            if val is slider_domain:
                regenerate_function()
                return
                
            # Check if eccentricity or c_factor changed, if so regenerate function tensor
            if val is slider_eccentricity or val is slider_c_factor:
                # Update the title of the plots to reflect new parameters
                axs[0].set_title(f'Current State (e={eccentricity:.1f}, c={c_factor:.1f})')
                fig.text(0.5, 0.93, f'Left: Current State | Right: Warped State (e={eccentricity:.1f}, c={c_factor:.1f})', 
                        ha='center', va='center', fontsize=10)
                
                # Update the FunctionTensor parameters
                function_tensor.eccentricity = eccentricity
                function_tensor.c_factor = c_factor
                
                # Recreate the grid and function values
                function_tensor._create_grid()
                function_tensor._compute_function_values()
                
                # Update current state visualization
                current_im.set_array(function_tensor.function_tensor.cpu().numpy())
                current_im.set_clim(function_tensor.function_tensor.min().item(), 
                                   function_tensor.function_tensor.max().item())
                
                # Update warped state visualization
                scaled_coords = normalized_coords_torch * domain_size
                warped_coords = function_tensor.warp_coordinates(
                    scaled_coords, 
                    r=slider_radius.val,
                    rotation_angles=[slider_theta_x.val, slider_theta_y.val, slider_theta_z.val]
                )
                warped_values = function_tensor.sample_at_ball_coords(
                    flat_to_ball(warped_coords, eccentricity=eccentricity, c_factor=c_factor)
                )
                warped_data = warped_values.reshape(resolution, resolution)
                
                warped_im.set_array(warped_data.cpu().numpy())
                warped_im.set_clim(warped_data.min().item(), warped_data.max().item())
                
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
    slider_eccentricity.on_changed(update)
    slider_c_factor.on_changed(update)
    slider_theta_x.on_changed(update)
    slider_theta_y.on_changed(update)
    slider_theta_z.on_changed(update)
    slider_anim_speed.on_changed(update)
    slider_flow.on_changed(update)  
    button_anim_toggle.on_clicked(toggle_animation)
    button_regen.on_clicked(regenerate_function)
    button_reset_state.on_clicked(reset_state)
    button_normalize_now.on_clicked(normalize_now)
    
    # Initial update
    update()
    
    plt.tight_layout(rect=[0, 0.45, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    main()
