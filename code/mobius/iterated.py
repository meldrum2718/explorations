import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import time

# Create a figure with a specific layout for visualization and controls
plt.style.use('dark_background')
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, height_ratios=[4, 1], width_ratios=[1, 1])

# Main plot areas
ax_main = fig.add_subplot(gs[0, :])  # Main visualization
ax_f1 = fig.add_subplot(gs[1, 0])    # First Möbius transformation
ax_f2 = fig.add_subplot(gs[1, 1])    # Second Möbius transformation

plt.subplots_adjust(bottom=0.25, hspace=0.3)

# Create a complex meshgrid
resolution = 800
x_min, x_max = -2, 2
y_min, y_max = -2, 2
x = np.linspace(x_min, x_max, resolution)
y = np.linspace(y_min, y_max, resolution)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Initial parameters for the two Möbius transformations
params1 = {'a': 0.8 - 0.2j, 'b': 0.1 + 0.2j, 'c': -0.1 + 0.1j, 'd': 0.9 - 0.1j}
params2 = {'a': 0.7 + 0.2j, 'b': -0.2 - 0.1j, 'c': 0.2 - 0.2j, 'd': 0.8 + 0.1j}

# Function to apply a Möbius transformation
def mobius_transform(z, a, b, c, d):
    denominator = c * z + d
    # Handle potential division by zero
    mask = np.abs(denominator) < 1e-10
    result = np.zeros_like(z, dtype=complex)
    result[~mask] = (a * z[~mask] + b) / denominator[~mask]
    result[mask] = np.inf  # Set to infinity where denominator is zero
    return result

# Function to get fixed points of a Möbius transformation
def get_fixed_points(a, b, c, d):
    if np.abs(c) < 1e-10:  # c ≈ 0
        if np.abs(a - d) < 1e-10:  # a ≈ d
            return [np.inf]  # Identity transformation
        return [b / (d - a), np.inf]
    
    # Solve quadratic equation: cz² + (d-a)z - b = 0
    discriminant = (d - a)**2 + 4*b*c
    if np.abs(discriminant) < 1e-10:  # Single fixed point
        return [-0.5 * (d - a) / c]
    
    z1 = (-1 * (d - a) + np.sqrt(discriminant)) / (2 * c)
    z2 = (-1 * (d - a) - np.sqrt(discriminant)) / (2 * c)
    return [z1, z2]

# Function to compute the escape time for each point
def compute_escape_time(z_grid, max_iter=30):
    z = z_grid.copy()
    escape_time = np.zeros(z.shape, dtype=int)
    mask = np.ones(z.shape, dtype=bool)
    
    for i in range(max_iter):
        # Apply first transformation
        z[mask] = mobius_transform(z[mask], 
                                  params1['a'], params1['b'], 
                                  params1['c'], params1['d'])
        
        # Check if points escaped or went to infinity
        new_mask = (np.abs(z) < 100) & np.isfinite(z)
        escaped = mask & ~new_mask
        escape_time[escaped] = i * 2 + 1
        mask = new_mask
        
        # Apply second transformation
        z[mask] = mobius_transform(z[mask], 
                                  params2['a'], params2['b'], 
                                  params2['c'], params2['d'])
        
        # Check again
        new_mask = (np.abs(z) < 100) & np.isfinite(z)
        escaped = mask & ~new_mask
        escape_time[escaped] = i * 2 + 2
        mask = new_mask
    
    # Points that never escaped
    escape_time[mask] = 0
    
    return escape_time, z

# Define a custom colormap for visualization
colors_list = [(0, 0, 0.4), (0, 0, 0.8), (0, 0.4, 1), 
               (0, 0.8, 1), (0.4, 1, 0.8), (0.8, 1, 0.4),
               (1, 0.8, 0), (1, 0.4, 0), (1, 0, 0), (0.4, 0, 0)]
cmap = LinearSegmentedColormap.from_list('mobius_cmap', colors_list, N=256)

# Compute initial escape time
escape_time, final_z = compute_escape_time(Z)

# Create the visualization
main_img = ax_main.imshow(
    escape_time,
    extent=[x_min, x_max, y_min, y_max],
    origin='lower',
    aspect='auto',
    cmap=cmap
)
ax_main.set_title('Julia-like Pattern from Alternating Möbius Transformations')
ax_main.set_xlabel('Re(z)')
ax_main.set_ylabel('Im(z)')

# Display fixed points on the plot
def plot_fixed_points():
    # Clear previous fixed points
    for ax in [ax_main, ax_f1, ax_f2]:
        for artist in ax.findobj(match=lambda x: hasattr(x, '_fixed_point_marker')):
            artist.remove()
    
    # Get fixed points
    fixed1 = get_fixed_points(params1['a'], params1['b'], params1['c'], params1['d'])
    fixed2 = get_fixed_points(params2['a'], params2['b'], params2['c'], params2['d'])
    
    # Plot fixed points on main plot
    for f in fixed1:
        if np.isfinite(f) and x_min <= f.real <= x_max and y_min <= f.imag <= y_max:
            p = ax_main.plot(f.real, f.imag, 'o', color='lime', markersize=8)[0]
            p._fixed_point_marker = True
    
    for f in fixed2:
        if np.isfinite(f) and x_min <= f.real <= x_max and y_min <= f.imag <= y_max:
            p = ax_main.plot(f.real, f.imag, 'o', color='red', markersize=8)[0]
            p._fixed_point_marker = True
    
    # Show fixed points for each transformation
    for f in fixed1:
        if np.isfinite(f):
            p = ax_f1.plot(f.real, f.imag, 'o', color='lime', markersize=6)[0]
            p._fixed_point_marker = True
    
    for f in fixed2:
        if np.isfinite(f):
            p = ax_f2.plot(f.real, f.imag, 'o', color='red', markersize=6)[0]
            p._fixed_point_marker = True

# Initial fixed points
plot_fixed_points()

# Add smaller plots for individual transformations
def visualize_single_mobius(ax, params, title):
    # Clear previous content
    ax.clear()
    
    # Create a meshgrid for the single transformation
    res = 100
    xs = np.linspace(-2, 2, res)
    ys = np.linspace(-2, 2, res)
    XS, YS = np.meshgrid(xs, ys)
    ZS = XS + 1j * YS
    
    # Apply transformation
    transformed = mobius_transform(ZS, params['a'], params['b'], params['c'], params['d'])
    
    # Create a phase visualization
    phase = np.angle(transformed)
    mag = np.abs(transformed)
    mag_normalized = 2 * np.arctan(mag) / np.pi  # Map magnitude to [0, 1]
    
    # Create HSV image
    h = (phase + np.pi) / (2 * np.pi)
    s = np.ones_like(h)
    v = mag_normalized
    
    # Handle infinities/NaNs
    mask = ~np.isfinite(transformed)
    h[mask] = 0
    s[mask] = 0
    v[mask] = 0.1
    
    # Stack and convert to RGB
    hsv = np.stack([h, s, v], axis=2)
    rgb = plt.cm.hsv(h)
    rgb[..., 3] = v  # Use magnitude for alpha
    
    # Display
    ax.imshow(rgb, extent=[-2, 2, -2, 2], origin='lower', aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')

# Create slider axes
slider_axes = []
for i, (param, value) in enumerate([
    ('a1_real', params1['a'].real), ('a1_imag', params1['a'].imag),
    ('b1_real', params1['b'].real), ('b1_imag', params1['b'].imag),
    ('c1_real', params1['c'].real), ('c1_imag', params1['c'].imag),
    ('d1_real', params1['d'].real), ('d1_imag', params1['d'].imag),
    ('a2_real', params2['a'].real), ('a2_imag', params2['a'].imag),
    ('b2_real', params2['b'].real), ('b2_imag', params2['b'].imag),
    ('c2_real', params2['c'].real), ('c2_imag', params2['c'].imag),
    ('d2_real', params2['d'].real), ('d2_imag', params2['d'].imag),
]):
    ax = plt.axes([0.1 + (i // 8) * 0.45, 0.02 + (7 - (i % 8)) * 0.025, 0.35, 0.02])
    slider_axes.append((param, ax))

# Create sliders
sliders = {}
for param, ax in slider_axes:
    # Extract the parameter key (a, b, c, or d) and which transformation (1 or 2)
    param_key = param[0]
    transform_num = param[1]
    component = 'real' if 'real' in param else 'imag'
    
    # Select the appropriate params dictionary
    params_dict = params1 if transform_num == '1' else params2
    
    # Get the initial value
    if component == 'real':
        init_val = params_dict[param_key].real
    else:
        init_val = params_dict[param_key].imag
        
    sliders[param] = Slider(ax, param, -1.5, 1.5, valinit=init_val)

# Function to update everything
def update(val):
    # Update transformation parameters
    params1['a'] = complex(sliders['a1_real'].val, sliders['a1_imag'].val)
    params1['b'] = complex(sliders['b1_real'].val, sliders['b1_imag'].val)
    params1['c'] = complex(sliders['c1_real'].val, sliders['c1_imag'].val)
    params1['d'] = complex(sliders['d1_real'].val, sliders['d1_imag'].val)
    
    params2['a'] = complex(sliders['a2_real'].val, sliders['a2_imag'].val)
    params2['b'] = complex(sliders['b2_real'].val, sliders['b2_imag'].val)
    params2['c'] = complex(sliders['c2_real'].val, sliders['c2_imag'].val)
    params2['d'] = complex(sliders['d2_real'].val, sliders['d2_imag'].val)
    
    # Update fixed points
    plot_fixed_points()
    
    # Update individual transformation visualizations
    visualize_single_mobius(ax_f1, params1, 'Transformation 1')
    visualize_single_mobius(ax_f2, params2, 'Transformation 2')
    
    # Compute new escape time
    escape_time, final_z = compute_escape_time(Z)
    main_img.set_data(escape_time)
    
    # Update the display
    fig.canvas.draw_idle()

# Connect sliders to update function
for slider in sliders.values():
    slider.on_changed(update)

# Initialize transformations display
visualize_single_mobius(ax_f1, params1, 'Transformation 1')
visualize_single_mobius(ax_f2, params2, 'Transformation 2')

# Add a reset button
reset_ax = plt.axes([0.8, 0.02, 0.1, 0.04])
reset_button = Button(reset_ax, 'Reset')

def reset(event):
    for slider in sliders.values():
        slider.reset()
    update(None)

reset_button.on_clicked(reset)

# Set a main plot title with explanation
ax_main.set_title('Julia-like Pattern from Alternating Möbius Transformations\n'
                 'Colors show escape time | Green dots: fixed points of T1 | Red dots: fixed points of T2')

plt.show()
