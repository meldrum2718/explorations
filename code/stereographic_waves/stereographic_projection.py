
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation

def warp(f, r, Rot, resolution=100, domain=(-5, 5)):
    """
    Visualize a function under stereographic projection and rotation.
    
    Parameters:
    -----------
    f : callable
        A function that maps R^2 -> R
    r : float
        Radius parameter for the stereographic projection
    Rot : numpy.ndarray
        3x3 rotation matrix
    resolution : int
        Number of points in each dimension of the mesh
    domain : tuple
        (min, max) values for both x and y coordinates
    
    Returns:
    --------
    numpy.ndarray
        The resulting image after projection, rotation, and function sampling
    """
    # Create meshgrid
    x_min, x_max = domain
    y_min, y_max = domain
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Flatten to list of vectors
    x_flat = X.flatten()
    y_flat = Y.flatten()
    
    # Step 1: Map each point (x,y) to a point on the sphere
    denom = x_flat**2 + y_flat**2 + r**2
    X_sphere = 2 * r**2 * x_flat / denom
    Y_sphere = 2 * r**2 * y_flat / denom
    Z_sphere = r * (x_flat**2 + y_flat**2 - r**2) / denom
    
    # Step 2: Apply rotation
    points = np.vstack((X_sphere, Y_sphere, Z_sphere)).T  # Shape: (n_points, 3)
    rotated_points = np.dot(points, Rot.T)  # Apply rotation
    
    # Step 3: Stereographic projection back to plane
    X_rot, Y_rot, Z_rot = rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2]
    
    # Handle points where Z_rot == 1 (division by zero)
    mask = np.abs(1 - Z_rot) > 1e-10
    X_proj = np.zeros_like(X_rot)
    Y_proj = np.zeros_like(Y_rot)
    
    X_proj[mask] = X_rot[mask] / (1 - Z_rot[mask])
    Y_proj[mask] = Y_rot[mask] / (1 - Z_rot[mask])
    
    # Sample function at projected points
    result = np.zeros_like(x_flat)
    result[mask] = f(X_proj[mask], Y_proj[mask])
    
    # Reshape to original mesh shape
    return result.reshape(X.shape)

# Example usage
def example_function(x, y):
    """Example function to visualize: a simple sinusoidal pattern"""
    return np.sin(x**2 + y**2) / (1 + 0.5*(x**2 + y**2))

# Create rotation matrices
def rotation_x(theta):
    """Rotation matrix around x-axis"""
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def rotation_y(theta):
    """Rotation matrix around y-axis"""
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def rotation_z(theta):
    """Rotation matrix around z-axis"""
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

# Create a figure with multiple visualizations
def plot_warped_function(f, r=1.0, angles=None, resolution=100, domain=(-5, 5)):
    """
    Plot the function under different rotations.
    
    Parameters:
    -----------
    f : callable
        A function that maps R^2 -> R
    r : float
        Radius parameter
    angles : list of tuples
        List of (x_angle, y_angle, z_angle) for rotations
    resolution : int
        Resolution of the meshgrid
    domain : tuple
        (min, max) values for both x and y coordinates
    """
    if angles is None:
        angles = [(0, 0, 0), (np.pi/4, 0, 0), (0, np.pi/4, 0), (0, 0, np.pi/4)]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (ax_angle, ay_angle, az_angle) in enumerate(angles):
        # Combine rotations
        Rot = np.dot(rotation_z(az_angle), 
                     np.dot(rotation_y(ay_angle), rotation_x(ax_angle)))
        
        # Apply warp function
        result = warp(f, r, Rot, resolution=resolution, domain=domain)
        
        # Plot
        im = axes[i].imshow(result, cmap='viridis', extent=[domain[0], domain[1], domain[0], domain[1]])
        axes[i].set_title(f'Rotation: ({ax_angle:.2f}, {ay_angle:.2f}, {az_angle:.2f})')
        fig.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    return fig

# Interactive visualization with sliders
def interactive_visualization(f, r=1.0, resolution=100, domain=(-5, 5)):
    """
    Create an interactive visualization with sliders to control rotation angles.
    
    Parameters:
    -----------
    f : callable
        A function that maps R^2 -> R
    r : float
        Radius parameter
    resolution : int
        Resolution of the meshgrid
    domain : tuple
        (min, max) values for both x and y coordinates
    """
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(left=0.1, bottom=0.25)
    
    # Initial rotation angles
    init_angles = (0, 0, 0)
    
    # Compute initial image
    Rot = np.dot(rotation_z(init_angles[2]), 
                np.dot(rotation_y(init_angles[1]), rotation_x(init_angles[0])))
    result = warp(f, r, Rot, resolution=resolution, domain=domain)
    
    # Display the image
    img = ax.imshow(result, cmap='viridis', extent=[domain[0], domain[1], domain[0], domain[1]])
    fig.colorbar(img, ax=ax)
    ax.set_title(f'Stereographic Projection with Rotation')
    
    # Create axes for sliders
    ax_x = plt.axes([0.1, 0.15, 0.8, 0.03])
    ax_y = plt.axes([0.1, 0.1, 0.8, 0.03])
    ax_z = plt.axes([0.1, 0.05, 0.8, 0.03])
    
    # Create sliders
    slider_x = Slider(ax_x, 'X Rotation', 0, 2*np.pi, valinit=init_angles[0], valstep=0.1)
    slider_y = Slider(ax_y, 'Y Rotation', 0, 2*np.pi, valinit=init_angles[1], valstep=0.1)
    slider_z = Slider(ax_z, 'Z Rotation', 0, 2*np.pi, valinit=init_angles[2], valstep=0.1)
    
    # Update function for sliders
    def update(_):
        ax_angle = slider_x.val
        ay_angle = slider_y.val
        az_angle = slider_z.val
        
        # Combine rotations
        Rot = np.dot(rotation_z(az_angle), 
                     np.dot(rotation_y(ay_angle), rotation_x(ax_angle)))
        
        # Apply warp function
        result = warp(f, r, Rot, resolution=resolution, domain=domain)
        
        # Update image
        img.set_data(result)
        fig.canvas.draw_idle()
    
    # Register the update function with each slider
    slider_x.on_changed(update)
    slider_y.on_changed(update)
    slider_z.on_changed(update)
    
    return fig, [slider_x, slider_y, slider_z]

# Animation with FuncAnimation
def animated_rotation(f, r=1.0, resolution=100, domain=(-5, 5), frames=100, interval=50):
    """
    Create an animation of rotating the stereographic projection.
    
    Parameters:
    -----------
    f : callable
        A function that maps R^2 -> R
    r : float
        Radius parameter
    resolution : int
        Resolution of the meshgrid
    domain : tuple
        (min, max) values for both x and y coordinates
    frames : int
        Number of frames in the animation
    interval : int
        Interval between frames in milliseconds
    """
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Initial rotation angles
    init_angles = (0, 0, 0)
    
    # Compute initial image
    Rot = np.dot(rotation_z(init_angles[2]), 
                np.dot(rotation_y(init_angles[1]), rotation_x(init_angles[0])))
    result = warp(f, r, Rot, resolution=resolution, domain=domain)
    
    # Display the image
    img = ax.imshow(result, cmap='viridis', extent=[domain[0], domain[1], domain[0], domain[1]])
    fig.colorbar(img, ax=ax)
    ax.set_title(f'Animated Stereographic Projection')
    
    # Animation update function
    def update(frame):
        # Calculate rotation angles based on frame
        ax_angle = frame * 2 * np.pi / frames
        ay_angle = frame * np.pi / frames
        az_angle = frame * np.pi / 2 / frames
        
        # Combine rotations
        Rot = np.dot(rotation_z(az_angle), 
                     np.dot(rotation_y(ay_angle), rotation_x(ax_angle)))
        
        # Apply warp function
        result = warp(f, r, Rot, resolution=resolution, domain=domain)
        
        # Update image
        img.set_data(result)
        return img,
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
    return fig, ani

# Visualize on the sphere (optional)
def plot_on_sphere(f, r=1.0, resolution=50):
    theta = np.linspace(0, 2*np.pi, resolution)
    phi = np.linspace(0, np.pi, resolution)
    
    # Create meshgrid for spherical coordinates
    THETA, PHI = np.meshgrid(theta, phi)
    
    # Convert to Cartesian coordinates
    X = r * np.sin(PHI) * np.cos(THETA)
    Y = r * np.sin(PHI) * np.sin(THETA)
    Z = r * np.cos(PHI)
    
    # Sample function value (using stereographic projection)
    # Map sphere points to plane
    x_plane = X / (r - Z)
    y_plane = Y / (r - Z)
    
    # Sample function
    values = f(x_plane, y_plane)
    
    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, facecolors=cm.viridis(values), alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Function visualized on sphere')
    
    plt.tight_layout()
    return fig

# Example usage
if __name__ == "__main__":
    # Define a more complex example function
    def complex_function(x, y):
        """A more interesting function to visualize"""
        return np.sin(x*y) * np.exp(-(x**2 + y**2)/20)
    
    # Create interactive visualization with sliders
    fig_interactive, sliders = interactive_visualization(complex_function, r=1.0, resolution=150)
    
    # # Create animation (uncomment to use)
    # fig_anim, animation = animated_rotation(complex_function, r=1.0, resolution=150)
    # # Save animation (uncomment to save)
    # # animation.save('stereographic_animation.gif', writer='pillow', fps=15)
    
    plt.show()
