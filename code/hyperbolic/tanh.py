import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from mpl_toolkits.mplot3d import Axes3D

def tanh_scaling(points, scale=1.0):
    """
    Apply tanh scaling to radial distance of points
    
    Parameters:
    -----------
    points : torch.Tensor
        Input points of shape (N, 2)
    scale : float
        Scaling factor for tanh function
        
    Returns:
    --------
    torch.Tensor
        Scaled points with same shape as input
    """
    # Calculate radial distance from origin
    r = torch.sqrt(torch.sum(points**2, dim=1))
    
    # Apply tanh scaling to the radius
    r_scaled = torch.tanh(r / scale) * scale
    
    # Avoid division by zero
    scaling_factor = torch.ones_like(r)
    mask = r > 1e-8
    scaling_factor[mask] = r_scaled[mask] / r[mask]
    
    # Scale the points
    return points * scaling_factor.unsqueeze(1)

def create_simplified_visualization(function_dict, resolution=100, scale=1.0, x_range=(-5, 5), y_range=(-5, 5)):
    """
    Create an interactive visualization of tanh scaling
    
    Parameters:
    -----------
    function_dict : dict
        Dictionary mapping function names to function objects
    resolution : int
        Grid resolution
    scale : float
        Scaling factor for tanh function
    x_range, y_range : tuple
        Range of x and y values
    """
    # Create meshgrid
    x = torch.linspace(x_range[0], x_range[1], resolution)
    y = torch.linspace(y_range[0], y_range[1], resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Flatten the grid points
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 9))
    
    # Original function plot
    ax1 = fig.add_subplot(131)
    
    # Transformed function plot
    ax3 = fig.add_subplot(133)
    
    # Initial function
    function_names = list(function_dict.keys())
    current_function = function_dict[function_names[0]]
    
    # Create widgets for interaction
    plt.subplots_adjust(bottom=0.25)
    
    ax_scale = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_func = plt.axes([0.25, 0.05, 0.65, 0.05])
    
    # Create sliders and radio buttons
    slider_scale = Slider(ax_scale, 'Scale Factor', 0.5, 5.0, valinit=scale, valstep=0.1)
    radio = RadioButtons(ax_func, function_names)
    
    # Sample the original function
    Z_orig = current_function(X.numpy(), Y.numpy())
    
    # Apply tanh scaling
    bounded_points = tanh_scaling(points, scale)
    
    # Reshape for visualization
    X_bounded = bounded_points[:, 0].reshape(resolution, resolution).numpy()
    Y_bounded = bounded_points[:, 1].reshape(resolution, resolution).numpy()
    
    # Calculate Z for 3D visualization - just the height of the 2D function
    Z_bounded = np.zeros_like(X_bounded)
    
    # Sample the function at the bounded points
    Z_transformed = current_function(X_bounded, Y_bounded)
    
    # Plot original function
    im1 = ax1.imshow(Z_orig, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], 
                    origin='lower', cmap='viridis')
    cbar1 = fig.colorbar(im1, ax=ax1)
    ax1.set_title('Original Function (Unbounded Space)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # Add circle to show boundary in original plot at tanh(1)
    circle1 = plt.Circle((0, 0), scale, fill=False, color='red', linestyle='--')
    ax1.add_patch(circle1)
    
    # Plot the bounded points in 3D
    Z_values = current_function(X_bounded, Y_bounded)
    
    
    
    # Plot transformed function
    im3 = ax3.imshow(Z_transformed, 
                    extent=[-scale, scale, -scale, scale], 
                    origin='lower', cmap='viridis')
    cbar3 = fig.colorbar(im3, ax=ax3)
    ax3.set_title('Function in Bounded Space')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    
    # Add circle in the transformed view
    circle3 = plt.Circle((0, 0), scale, fill=False, color='red', linestyle='--')
    ax3.add_patch(circle3)
    
    # Update function for slider and radio buttons
    def update(_):
        # Get current scale
        scale_val = slider_scale.val
        
        # Get current function
        current_func = function_dict[radio.value_selected]
        
        # Update original function
        Z_orig = current_func(X.numpy(), Y.numpy())
        im1.set_array(Z_orig)
        im1.set_clim(Z_orig.min(), Z_orig.max())
        
        # Remove old circle and add new one
        for patch in ax1.patches:
            patch.remove()
        circle1 = plt.Circle((0, 0), scale_val, fill=False, color='red', linestyle='--')
        ax1.add_patch(circle1)
        
        # Apply tanh scaling with new scale
        bounded_points = tanh_scaling(points, scale_val)
        
        # Reshape for visualization
        X_bounded = bounded_points[:, 0].reshape(resolution, resolution).numpy()
        Y_bounded = bounded_points[:, 1].reshape(resolution, resolution).numpy()
        Z_bounded = np.zeros_like(X_bounded)
        
        # Sample function at bounded points
        Z_values = current_func(X_bounded, Y_bounded)
        
        # Update transformed function
        Z_transformed = current_func(X_bounded, Y_bounded)
        im3.set_array(Z_transformed)
        im3.set_clim(Z_transformed.min(), Z_transformed.max())
        im3.set_extent([-scale_val, scale_val, -scale_val, scale_val])
        
        # Remove old circle and add new one in transformed view
        for patch in ax3.patches:
            patch.remove()
        circle3 = plt.Circle((0, 0), scale_val, fill=False, color='red', linestyle='--')
        ax3.add_patch(circle3)
        
        # Redraw
        fig.canvas.draw_idle()
    
    # Register update function with slider and radio buttons
    slider_scale.on_changed(update)
    radio.on_clicked(update)
    
    plt.tight_layout()
    return fig, [slider_scale, radio]

# Example functions to visualize (same as original)
def periodic_waves(x, y):
    """Periodic wave pattern over R²"""
    return np.sin(x) * np.sin(y)

def periodic_checkerboard(x, y):
    """Periodic checkerboard pattern over R²"""
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def radial_rings(x, y):
    """Simple radial rings that extend infinitely"""
    r = np.sqrt(x**2 + y**2)
    return np.sin(r)

def complex_periodic(x, y):
    """Complex periodic pattern with multiple frequencies"""
    pattern1 = np.sin(x) * np.sin(y)
    pattern2 = np.sin(2*x) * np.sin(2*y)
    pattern3 = np.sin(0.5*x + 0.5*y)
    
    return (pattern1 + 0.5*pattern2 + 0.3*pattern3) / 1.8

# Main execution
if __name__ == "__main__":
    # Define functions to visualize
    function_dict = {
        'periodic_waves': periodic_waves,
        'periodic_checkerboard': periodic_checkerboard,
        'radial_rings': radial_rings,
        'complex_periodic': complex_periodic
    }
    
    # Create the interactive visualization
    fig, widgets = create_simplified_visualization(function_dict, resolution=100, scale=1.0)
    
    plt.show()
