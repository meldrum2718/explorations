import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from typing import Optional, Callable, Tuple
import math

from coordinate_transformations import create_stereographic_transform
from hyperbolic_function_tensor import HyperbolicFunctionTensor

# =============================================================================
# Demo Function and Interactive Interface
# =============================================================================

def demo_function(points: torch.Tensor) -> torch.Tensor:
    """Demo function: sin(x) + sin(y)"""
    x, y = points[:, 0], points[:, 1]
    values = torch.sin(x) + torch.sin(y)
    return values.unsqueeze(1)


def main():
    """Interactive demo with stereographic projection controls."""
    # Device selection
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    
    # Setup figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(bottom=0.45)
    
    # Initialize parameters for 2D (3D rotations on sphere)
    # For 2D->3D stereographic, we have 3 rotation planes: xy, xz, yz
    params = {
        'radius': 1.0,
        'angle_xy': 0.0,  # Rotation in xy plane
        'angle_xz': 0.0,  # Rotation in xz plane  
        'angle_yz': 0.0,  # Rotation in yz plane
        'resolution': 32
    }
    
    def update_plot():
        """Update visualization with current parameter values."""
        # Store current zoom levels (only if plots have been initialized)
        try:
            ax1_xlim = ax1.get_xlim()
            ax1_ylim = ax1.get_ylim()
            ax2_xlim = ax2.get_xlim()
            ax2_ylim = ax2.get_ylim()
            # Check if this is the first plot (default matplotlib limits)
            first_plot = (ax1_xlim == (0.0, 1.0) and ax1_ylim == (0.0, 1.0))
        except:
            # First plot - no limits to preserve
            first_plot = True
        
        # Create rotation angles list for 3D space (2D input -> 3D sphere)
        rotation_angles = [params['angle_xy'], params['angle_xz'], params['angle_yz']]
        
        # Create stereographic transformation
        pullback_fn, pushforward_fn = create_stereographic_transform(
            radius=params['radius'],
            rotation_angles=rotation_angles,
            n_dims=2,
            device=device
        )
        
        # Create hyperbolic function
        hft = HyperbolicFunctionTensor.from_function(
            demo_function, 
            resolution=int(params['resolution']),
            n_dims=2, 
            n_channels=1,
            pullback_fn=pullback_fn, 
            pushforward_fn=pushforward_fn, 
            device=device
        )
        
        # Render images
        image_global = hft.render_2d(resolution=128, extent=3.0)
        image_ball = hft.values[:, :, 0]
        
        # Clear and update plots
        ax1.clear()
        ax2.clear()
        
        ax1.imshow(image_global.cpu().numpy(), extent=[-3, 3, -3, 3], 
                  origin='lower', cmap='viridis')
        ax1.set_title('Stereographic Hyperbolic Function')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.grid(True, alpha=0.3)
        
        ax2.imshow(image_ball.cpu().numpy(), extent=[-1, 1, -1, 1],
                  origin='lower', cmap='viridis')
        ax2.set_title('Ball Coordinates')
        ax2.set_xlabel('ball x')
        ax2.set_ylabel('ball y')
        ax2.grid(True, alpha=0.3)
        
        # Restore zoom levels if they were modified by user (but not on first plot)
        if not first_plot:
            if ax1_xlim != (-3.0, 3.0) or ax1_ylim != (-3.0, 3.0):
                ax1.set_xlim(ax1_xlim)
                ax1.set_ylim(ax1_ylim)
            
            if ax2_xlim != (-1.0, 1.0) or ax2_ylim != (-1.0, 1.0):
                ax2.set_xlim(ax2_xlim)
                ax2.set_ylim(ax2_ylim)
        
        # Display parameters
        param_text = (f"Stereographic Parameters:\n"
                     f"Radius: {params['radius']:.2f}\n"
                     f"Rotation XY: {params['angle_xy']:.2f}\n"
                     f"Rotation XZ: {params['angle_xz']:.2f}\n"
                     f"Rotation YZ: {params['angle_yz']:.2f}")
        ax1.text(0.02, 0.98, param_text, transform=ax1.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.draw()
    
    # Create sliders
    slider_configs = [
        ('radius', 'Radius', 0.1, 500.0),
        ('angle_xy', 'Rotation XY', 0, 2*math.pi),
        ('angle_xz', 'Rotation XZ', 0, 2*math.pi),
        ('angle_yz', 'Rotation YZ', 0, 2*math.pi),
        ('resolution', 'Resolution', 8, 1024)
    ]
    
    sliders = {}
    slider_height = 0.03
    slider_spacing = 0.04
    slider_width = 0.6
    start_y = 0.35
    
    for i, (param, label, vmin, vmax) in enumerate(slider_configs):
        y_pos = start_y - i * slider_spacing
        ax = plt.axes([0.2, y_pos, slider_width, slider_height])
        
        if param == 'resolution':
            slider = Slider(ax, label, vmin, vmax, valinit=params[param], valfmt='%d')
        else:
            slider = Slider(ax, label, vmin, vmax, valinit=params[param])
        sliders[param] = slider
    
    def update_params(val):
        """Update parameters from slider values."""
        for param, slider in sliders.items():
            params[param] = slider.val
        update_plot()
    
    # Connect sliders
    for slider in sliders.values():
        slider.on_changed(update_params)
    
    # Reset button
    ax_reset = plt.axes([0.05, 0.02, 0.1, 0.04])
    button_reset = Button(ax_reset, 'Reset')
    
    def reset_sliders(event):
        for slider in sliders.values():
            slider.reset()
    
    button_reset.on_clicked(reset_sliders)
    
    # Initial render
    update_plot()
    plt.show()


if __name__ == "__main__":
    main()
