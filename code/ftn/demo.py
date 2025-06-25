"""

in the middle of making this code work for high.dim function tensors
next step:
    include sliders for nd stereographic rotation
    also include possiblity of using GL(n, Z) for the warping
    make sure translation vector is in R^n


"""


import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation

from coordinate_transformations import create_stereographic_transform, warp_with_rotation, create_nd_rotation_matrix
from function_tensor import FunctionTensor, normalize
from function_tensor_network import FunctionTensorNetwork

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

def setup_sliders(fig, config, valinit_params, discrete_sliders=None):
    if discrete_sliders is None:
        discrete_sliders = []
    sliders = {}
    slider_height = 0.03
    slider_spacing = 0.04
    slider_width = 0.6
    start_y = 0.35
    
    for i, (param, label, vmin, vmax) in enumerate(config):
        y_pos = start_y - i * slider_spacing
        ax = fig.add_subplot([0.2, y_pos, slider_width, slider_height])
        
        if param in discrete_sliders:
            slider = Slider(ax, label, vmin, vmax, valinit=valinit_params[param], valfmt='%d', valstep=1)
        else:
            slider = Slider(ax, label, vmin, vmax, valinit=valinit_params[param])
        sliders[param] = slider

    return sliders


def get_pushforward(radius, rotation_matrix, translation):
    def pushforward(global_coords):
        n_dims = rotation_matrix.shape[-1] - 1
        warped_coords = warp_with_rotation(
            global_coords.reshape(-1, n_dims) + translation,
            r=radius,
            rotation_matrix=rotation_matrix
        ).reshape(global_coords.shape)
        return warped_coords
    return pushforward



def main():
    """Interactive demo with stereographic projection controls."""
    device = get_device()
    print(f"Using device: {device}")

    params = {
        'radius': 1.0,
        'angle_xy': 0.0,  # Rotation in xy plane
        'angle_xz': 0.0,  # Rotation in xz plane  
        'angle_yz': 0.0,  # Rotation in yz plane
        't_x': 0.0,  # Translation
        't_y': 0.0,  # Translation
        'resolution': 8,
        'blend_factor': 1,
    }


    channels = 1
    n_dims = 4
    resolution = params['resolution']
    ft = FunctionTensor(
        torch.rand(
            *((resolution,) * n_dims), channels
        )
    )

    # Setup figure
    fig, ax = plt.subplots(1, figsize=(14, 6))
    im = ax.imshow(ft.to_flat())
    plt.subplots_adjust(bottom=0.45)

    slider_config = [
        ('radius', 'Radius', 0.001, 10.0),
        ('angle_xy', 'Rotation XY', 0, 1),
        ('angle_xz', 'Rotation XZ', 0, 1),
        ('angle_yz', 'Rotation YZ', 0, 1),
        ('t_x', 'x translation', -3, 3),
        ('t_y', 'y translation', -3, 3),
        ('resolution', 'Resolution', 8, 1024),
        ('blend_factor', 'Blend Factor', 0, 1)
    ]

    # if use_lie_alg_like := True:
    rot_max = 0.1
    t_max = 0.1
    slider_config = [
        ('radius', 'Radius', 0.0010, 2.0),
        ('angle_xy', 'Rotation XY', -rot_max, rot_max),
        ('angle_xz', 'Rotation XZ', -rot_max, rot_max),
        ('angle_yz', 'Rotation YZ', -rot_max, rot_max),
        ('t_x', 'x translation', -t_max, t_max),
        ('t_y', 'y translation', -t_max, t_max),
        ('resolution', 'Resolution', 3, 1024),
        ('blend_factor', 'Blend Factor', 0, 1)
    ]
    
    sliders = setup_sliders(
        fig,
        slider_config,
        valinit_params=params,
        discrete_sliders=['resolution']
    )
    
    def update_params(val):
        """Update parameters from slider values."""
        for param, slider in sliders.items():
            params[param] = slider.val
        # update_plot()
    
    # Connect sliders
    for slider in sliders.values():
        slider.on_changed(update_params)
    
    # Reset button
    ax_reset = plt.axes([0.05, 0.02, 0.1, 0.04])
    button_reset = Button(ax_reset, 'Reset')
    
    def reset_sliders(event):
        for slider in sliders.values():
            slider.reset()
        sliders['blend_factor'].val = 1 ## TODO need this explicitly ?
    
    button_reset.on_clicked(reset_sliders)

    def update_plot(frame=None):
        """Update visualization with current parameter values."""
        nonlocal ft

        if params['resolution'] != ft.resolution:
            ft = ft.resample(params['resolution'])
        
        # Create rotation angles list for 3D space (2D input -> 3D sphere)
        rotation_angles = [
            2 * torch.pi * params['angle_xy'],
            2 * torch.pi * params['angle_xz'],
            2 * torch.pi * params['angle_yz']
        ]
        
        rotation_matrix = create_nd_rotation_matrix(angles=rotation_angles, dim=ft.n_dims+1)
        translation = torch.Tensor([0, 0, params['t_x'], params['t_y']]) ## TODO 
        pushforward = get_pushforward(radius=params['radius'], rotation_matrix=rotation_matrix, translation=translation)
        mesh = FunctionTensor.generate_global_mesh_coords(resolution=params['resolution'], n_dims=ft.n_dims)
        new_state = ft(pushforward(mesh))
        ft.blend(new_state=new_state, blend_factor=params['blend_factor'])

        
        im.set_data(ft.to_flat())

        plt.draw()
        # return [current_im, warped_im] ## TODO define the images outside of update, do im.set_data. no ax.clear
    
    update_plot()
    ani = FuncAnimation(
        fig=fig,
        func=update_plot,
        interval=30,
        cache_frame_data=False,
        # blit=True,
        # save_count=1,
    )

    plt.show()



if __name__ == "__main__":
    main()
