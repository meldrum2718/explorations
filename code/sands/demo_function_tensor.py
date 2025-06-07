import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from typing import Optional, Callable, Tuple
import math

from coordinate_transformations import create_stereographic_transform, warp_with_rotation, create_nd_rotation_matrix
from function_tensor import FunctionTensor, normalize

# =============================================================================
# Demo Function and Interactive Interface
# =============================================================================

def demo_function(points: torch.Tensor) -> torch.Tensor:
    """ Demo function: sin(pi*x) + sin(pi*y) """
    x, y = points[:, 0], points[:, 1]
    values = torch.sin(torch.pi * x) + torch.sin(torch.pi * y)
    return values.unsqueeze(1)

def get_device():
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    return device


def main():
    """Interactive demo with stereographic projection controls."""
    device = get_device()
    print(f"Using device: {device}")

    params = {
        'radius': 1.0,
        'angle_xy': 0.0,  # Rotation in xy plane
        'angle_xz': 0.0,  # Rotation in xz plane  
        'angle_yz': 0.0,  # Rotation in yz plane
        'resolution': 1024,
        'blend_factor': 0,
    }

    print('making function tensor')
    ft = FunctionTensor.from_function(
        func=demo_function, 
        resolution=int(params['resolution']),
        n_dims=2, 
        channels=1,
        # device=device
    )
    print('made function tensor')
    print('ft.ndims', ft.n_dims)
    
    # Setup figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(bottom=0.45)
    
    def update_plot():
        """Update visualization with current parameter values."""
        nonlocal ft

        # # Store current zoom levels (only if plots have been initialized)
        # try:
        #     ax1_xlim = ax1.get_xlim()
        #     ax1_ylim = ax1.get_ylim()
        #     ax2_xlim = ax2.get_xlim()
        #     ax2_ylim = ax2.get_ylim()
        #     # Check if this is the first plot (default matplotlib limits)
        #     first_plot = (ax1_xlim == (0.0, 1.0) and ax1_ylim == (0.0, 1.0))
        # except:
        #     # First plot - no limits to preserve
        #     first_plot = True
        
        # Create rotation angles list for 3D space (2D input -> 3D sphere)
        rotation_angles = [params['angle_xy'], params['angle_xz'], params['angle_yz']]
        
        # if params['resolution'] != ft.resolution:
        #     res_ratio = params['resolution'] / ft.resolution
        #     ft = ft.resample(params['resolution'])
        #     ax1_xlim = (ax1_xlim[0] * res_ratio, ax1_xlim[1] * res_ratio)
        #     ax1_ylim = (ax1_ylim[0] * res_ratio, ax1_ylim[1] * res_ratio)

        #     ax2_xlim = (ax2_xlim[0] * res_ratio, ax2_xlim[1] * res_ratio)
        #     ax2_ylim = (ax2_ylim[0] * res_ratio, ax2_ylim[1] * res_ratio)

        
        rotation_matrix = create_nd_rotation_matrix(angles=rotation_angles, dim=3)
        global_coords = FunctionTensor.generate_global_mesh_coords(params['resolution'], n_dims=2)
        warped_coords = warp_with_rotation(global_coords.reshape(-1, 2), r=params['radius'], rotation_matrix=rotation_matrix).reshape(global_coords.shape)

        print('- gc.s', global_coords.shape)
        print('- gc.s[:-1]', global_coords.shape[:-1])
        print('- gc.rs -1 2', global_coords.reshape(-1, 2).shape)
        # image = ft(global_coords)
        # print('>>>>>>> ------- mean diff:', torch.mean((image - ft.tensor).reshape(-1)))
        image = ft.tensor
        warped_image = ft(warped_coords)
        
        # Clear and update plots
        ax1.clear()
        ax2.clear()
        
        ax1.imshow(image.cpu().numpy(), origin='lower', cmap='viridis')
        ax1.set_title('Image')
        ax1.grid(False)
        
        ax2.imshow(warped_image.cpu().numpy(), origin='lower', cmap='viridis')
        ax2.set_title('Warped Image')
        ax2.grid(False)
        
        
        # Display parameters
        param_text = (f"Stereographic Parameters:\n"
                     f"Radius: {params['radius']:.2f}\n"
                     f"Rotation XY: {params['angle_xy']:.2f}\n"
                     f"Rotation XZ: {params['angle_xz']:.2f}\n"
                     f"Rotation YZ: {params['angle_yz']:.2f}")
        ax1.text(0.02, 0.98, param_text, transform=ax1.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # if not first_plot:
        #     ax1.set_xlim(ax1_xlim)
        #     ax1.set_ylim(ax1_ylim)
        #     ax2.set_xlim(ax2_xlim)
        #     ax2.set_ylim(ax2_ylim)
        
        plt.draw()

        ## update state
        t = params['blend_factor']
        new_state = (1 - t) * normalize(image) + t * normalize(warped_image)
        new_state = normalize(new_state)
        ft.tensor = new_state
        ft.resolution = params['resolution']

    
    # Create sliders
    slider_configs = [
        ('radius', 'Radius', 0.01, 10.0),
        ('angle_xy', 'Rotation XY', 0, 2*math.pi),
        ('angle_xz', 'Rotation XZ', 0, 2*math.pi),
        ('angle_yz', 'Rotation YZ', 0, 2*math.pi),
        ('resolution', 'Resolution', 8, 1024),
        ('blend_factor', 'Blend Factor', 0, 1)
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
            slider = Slider(ax, label, vmin, vmax, valinit=params[param], valfmt='%d', valstep=1)
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
