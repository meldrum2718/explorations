import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.animation import FuncAnimation

# Import the stereographic projection code
from stereographic_projection import warp, rotation_x, rotation_y, rotation_z

# Import the tensor wave simulation
from tensor_wave_simulation import TensorWaveSimulation, wave_tensor_field

def interactive_tensor_wave_demo():
    """
    Interactive demo combining tensor-based wave equation simulation with stereographic projection.
    Wave propagation happens directly on the density field using tensor operations.
    """
    # Enable PyTorch's default tensor type to be float32
    torch.set_default_dtype(torch.float32)
    
    # Setup parameters
    resolution = 200
    domain = (-5, 5)
    r = 1.0  # radius parameter for stereographic projection
    
    # Create the wave simulation with much faster wave propagation
    sim = TensorWaveSimulation(resolution=resolution, dt=0.1, c=0.8, domain=domain, damping=0.005)
    
    # Create figure and axes
    fig = plt.figure(figsize=(15, 8))
    
    # Layout with gridspec
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, height_ratios=[4, 1])
    
    # Two main visualization panels
    ax_density = fig.add_subplot(gs[0, 0])  # Density field with wave
    ax_projection = fig.add_subplot(gs[0, 1])  # Stereographic projection
    
    # Control panel at bottom
    ax_controls = fig.add_subplot(gs[1, :])
    ax_controls.axis('off')
    
    # Setup visualization initial state
    density_field = sim.get_density()
    density_img = ax_density.imshow(density_field, cmap='coolwarm', 
                                   extent=[domain[0], domain[1], domain[0], domain[1]])
    ax_density.set_title('Wave Field')
    
    # Add colorbar
    fig.colorbar(density_img, ax=ax_density, shrink=0.8)
    
    # Initial rotation
    init_angles = (0, 0, 0)
    Rot = np.dot(rotation_z(init_angles[2]), 
                np.dot(rotation_y(init_angles[1]), rotation_x(init_angles[0])))
    
    # Function to sample the density field
    def sample_density(x, y):
        return wave_tensor_field(x, y, sim, mode='density')
    
    # Initial projection
    projection = warp(sample_density, r, Rot, resolution=resolution, domain=domain)
    projection_img = ax_projection.imshow(projection, cmap='coolwarm',
                                        extent=[domain[0], domain[1], domain[0], domain[1]])
    ax_projection.set_title('Stereographic Projection')
    
    # Add colorbar for projection
    projection_cbar = fig.colorbar(projection_img, ax=ax_projection, shrink=0.8)
    
    # Create sliders for rotation - centered around zero
    slider_ax_x = plt.axes([0.1, 0.15, 0.8, 0.03])
    slider_ax_y = plt.axes([0.1, 0.1, 0.8, 0.03])
    slider_ax_z = plt.axes([0.1, 0.05, 0.8, 0.03])
    
    # Add wave speed and stereo weight sliders
    slider_wave_speed_ax = plt.axes([0.1, 0.20, 0.3, 0.03])
    slider_stereo_ax = plt.axes([0.6, 0.20, 0.3, 0.03])
    
    # Add radius and domain sliders
    slider_radius_ax = plt.axes([0.1, 0.25, 0.3, 0.03])
    slider_domain_ax = plt.axes([0.6, 0.25, 0.3, 0.03])
    
    # Center all sliders around zero with smaller range
    slider_x = Slider(slider_ax_x, 'X Rotation', -0.5, 0.5, valinit=0)
    slider_y = Slider(slider_ax_y, 'Y Rotation', -0.5, 0.5, valinit=0)
    slider_z = Slider(slider_ax_z, 'Z Rotation', -0.5, 0.5, valinit=0)
    
    # Create wave speed and stereo weight sliders
    slider_wave_speed = Slider(slider_wave_speed_ax, 'Wave Speed', 0.0, 2.0, valinit=0.0)
    slider_stereo = Slider(slider_stereo_ax, 'Stereo Weight', 0.0, 0.01, valinit=0.0)
    
    # Create radius and domain sliders
    slider_radius = Slider(slider_radius_ax, 'Radius', 0.01, 10.0, valinit=r)
    slider_domain = Slider(slider_domain_ax, 'Domain Size', 1.0, 10.0, valinit=domain[1])
    
    # Create buttons for controls - position them at the bottom
    btn_reset_ax = plt.axes([0.1, 0.35, 0.3, 0.05])
    btn_reset = Button(btn_reset_ax, 'Reset Wave')
    
    btn_anim_ax = plt.axes([0.6, 0.35, 0.3, 0.05])
    btn_anim = Button(btn_anim_ax, 'Start Wave Propagation')
    
    # Animation state
    anim = None
    rotating = False
    pulse_type = "center"  # Default pulse type
    
    # Add variables to track animation angles (separate from slider values)
    animation_angles = {'x': 0, 'y': 0, 'z': 0}
    
    # Update functions
    def update_rotation(_):
        """Update rotation based on slider values"""
        nonlocal projection
        ax_angle = slider_x.val
        ay_angle = slider_y.val
        az_angle = slider_z.val
        
        # Combine rotations
        Rot = np.dot(rotation_z(az_angle), 
                     np.dot(rotation_y(ay_angle), rotation_x(ax_angle)))
        
        # Set the rotation matrix in the simulation for stereographic blending
        sim.set_rotation_matrix(Rot)
        
        # Apply warp function with current sampling function
        projection = warp(sample_density, slider_radius.val, Rot, resolution=resolution, domain=(-slider_domain.val, slider_domain.val))
        
        # Update image
        projection_img.set_data(projection)
        
        # Update color limits
        vmin = np.min(density_field)
        vmax = np.max(density_field)
        projection_img.set_clim(vmin, vmax)
            
        fig.canvas.draw_idle()
    
    def update_domain(_):
        """Update domain size from slider"""
        nonlocal domain, projection
        # Update domain
        domain_size = slider_domain.val
        domain = (-domain_size, domain_size)
        
        # Update extent for both displays
        density_img.set_extent([domain[0], domain[1], domain[0], domain[1]])
        projection_img.set_extent([domain[0], domain[1], domain[0], domain[1]])
        
        # Update projection
        Rot = np.dot(rotation_z(slider_z.val), 
                    np.dot(rotation_y(slider_y.val), rotation_x(slider_x.val)))
                    
        projection = warp(sample_density, slider_radius.val, Rot, resolution=resolution, domain=domain)
        projection_img.set_data(projection)
        
        fig.canvas.draw_idle()
    
    def update_radius(_):
        """Update radius parameter for stereographic projection"""
        nonlocal projection
        # Update projection with new radius
        Rot = np.dot(rotation_z(slider_z.val), 
                    np.dot(rotation_y(slider_y.val), rotation_x(slider_x.val)))
                    
        projection = warp(sample_density, slider_radius.val, Rot, resolution=resolution, domain=domain)
        projection_img.set_data(projection)
        
        fig.canvas.draw_idle()
    
    def on_reset(_):
        """Reset the wave simulation to initial state"""
        sim.reset(pulse_type.lower())
        density_field = sim.get_density()
        density_img.set_data(density_field)
        
        # Reset animation angles
        animation_angles['x'] = 0
        animation_angles['y'] = 0
        animation_angles['z'] = 0
        
        update_rotation(None)  # Update projection
        fig.canvas.draw_idle()
    
    def on_animation(event):
        """Start/stop the wave propagation animation"""
        nonlocal anim, rotating
        
        if anim is not None:
            anim.event_source.stop()
            anim = None
            btn_anim.label.set_text('Start Wave Propagation')
            rotating = False
            return
        
        btn_anim.label.set_text('Stop Wave Propagation')
        rotating = True
        
        # Reset animation angles to continue from current slider values
        animation_angles['x'] = slider_x.val
        animation_angles['y'] = slider_y.val
        animation_angles['z'] = slider_z.val
        
        # Frame generator for rotation animation
        def frame_generator():
            frame = 0
            while True:
                yield frame
                frame += 1
        
        # Animation update function
        def update_frame(frame):
            # Update wave simulation multiple times per frame for faster animation
            for _ in range(3):  # Perform multiple simulation steps per frame
                density_field = sim.step()
            
            # Update density display
            density_img.set_data(density_field)
            
            # Adjust color scale if needed
            vmin = np.min(density_field)
            vmax = np.max(density_field)
            density_img.set_clim(vmin, vmax)
            
            if rotating:
                # Calculate rotation angles directly from sliders (no animation for rotation)
                ax_angle = slider_x.val
                ay_angle = slider_y.val
                az_angle = slider_z.val
                
                # Calculate rotation matrix directly without updating sliders
                Rot = np.dot(rotation_z(az_angle), 
                            np.dot(rotation_y(ay_angle), rotation_x(ax_angle)))
                
                # Set the rotation matrix in the simulation for stereographic blending
                sim.set_rotation_matrix(Rot)
                
                # Apply warp function with current sampling function using current radius and domain
                projection = warp(sample_density, slider_radius.val, Rot, 
                                 resolution=resolution, domain=(-slider_domain.val, slider_domain.val))
                
                # Update projection image
                projection_img.set_data(projection)
            
            return density_img, projection_img
        
        # Create animation with save_count=1 to suppress warnings
        anim = FuncAnimation(fig, update_frame, frames=frame_generator(), 
                            interval=50, blit=True, save_count=1, cache_frame_data=False)
    
    # Wave parameter update function
    def update_wave_params(_):
        """Update wave speed and stereo weight from sliders"""
        sim.c = slider_wave_speed.val
        sim.set_stereo_weight(slider_stereo.val)
    
    # Connect callbacks
    slider_x.on_changed(update_rotation)
    slider_y.on_changed(update_rotation)
    slider_z.on_changed(update_rotation)
    slider_wave_speed.on_changed(update_wave_params)
    slider_stereo.on_changed(update_wave_params)
    slider_radius.on_changed(update_radius)
    slider_domain.on_changed(update_domain)
    btn_reset.on_clicked(on_reset)
    btn_anim.on_clicked(on_animation)
    
    # Use subplots_adjust instead of tight_layout to avoid warnings
    plt.subplots_adjust(bottom=0.3, left=0.05, right=0.95, top=0.95)
    plt.show()

if __name__ == "__main__":
    interactive_tensor_wave_demo()
