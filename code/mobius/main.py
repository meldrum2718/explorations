
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import matplotlib.colors as colors

# Set up the figure and axes
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
plt.subplots_adjust(bottom=0.25)  # Make room for sliders

# Create a meshgrid for the complex plane
resolution = 500
x_min, x_max = -2, 2
y_min, y_max = -2, 2
x = np.linspace(x_min, x_max, resolution)
y = np.linspace(y_min, y_max, resolution)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Initial values for Möbius transformation parameters
a_init, b_init, c_init, d_init = 1, 0, 0, 1

# Function to apply Möbius transformation: f(z) = (az + b) / (cz + d)
def mobius_transform(z, a, b, c, d):
    # Handle division by zero (where cz + d = 0)
    denominator = c * z + d
    # Set a small value where denominator is zero to avoid division by zero
    denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
    return (a * z + b) / denominator

# Function to create a color map from complex values
def complex_to_rgb(z):
    # Use phase for hue and magnitude for brightness
    h = (np.angle(z) + np.pi) / (2 * np.pi)  # Map phase to [0, 1]
    s = np.ones_like(h)  # Full saturation with same shape as h
    v = 2 * np.arctan(np.abs(z)) / np.pi  # Map magnitude to [0, 1] using arctan
    
    # Convert HSV to RGB
    hsv = np.stack([h, s, v], axis=2)
    return colors.hsv_to_rgb(hsv)

# Initial transformation
Z_transformed = mobius_transform(Z, a_init, b_init, c_init, d_init)
img = ax.imshow(complex_to_rgb(Z_transformed), extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto')
plt.title('Möbius Transformation: (az + b) / (cz + d)')

# Create sliders
ax_a = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_b = plt.axes([0.25, 0.10, 0.65, 0.03])
ax_c = plt.axes([0.25, 0.05, 0.65, 0.03])
ax_d = plt.axes([0.25, 0.00, 0.65, 0.03])

slider_a = Slider(ax_a, 'a (real)', -10, 10, valinit=a_init)
slider_b = Slider(ax_b, 'b (real)', -10, 10, valinit=b_init)
slider_c = Slider(ax_c, 'c (real)', -10, 10, valinit=c_init)
slider_d = Slider(ax_d, 'd (real)', -10, 10, valinit=d_init)

# Update function
def update(val):
    a = complex(slider_a.val)
    b = complex(slider_b.val)
    c = complex(slider_c.val)
    d = complex(slider_d.val)
    
    Z_new = mobius_transform(Z, a, b, c, d)
    img.set_data(complex_to_rgb(Z_new))
    
    # Update title with current values
    plt.title(f'Möbius Transformation: ({a:.2f}z + {b:.2f}) / ({c:.2f}z + {d:.2f})')
    fig.canvas.draw_idle()

# Register the update function with each slider
slider_a.on_changed(update)
slider_b.on_changed(update)
slider_c.on_changed(update)
slider_d.on_changed(update)

# Add complex parameter support
ax_a_im = plt.axes([0.25, 0.20, 0.65, 0.03])
ax_b_im = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_c_im = plt.axes([0.25, 0.30, 0.65, 0.03])
ax_d_im = plt.axes([0.25, 0.35, 0.65, 0.03])

slider_a_im = Slider(ax_a_im, 'a (imag)', -2, 2, valinit=0)
slider_b_im = Slider(ax_b_im, 'b (imag)', -2, 2, valinit=0)
slider_c_im = Slider(ax_c_im, 'c (imag)', -2, 2, valinit=0)
slider_d_im = Slider(ax_d_im, 'd (imag)', -2, 2, valinit=0)

# Extended update function for complex parameters
def update_complex(val):
    a = complex(slider_a.val, slider_a_im.val)
    b = complex(slider_b.val, slider_b_im.val)
    c = complex(slider_c.val, slider_c_im.val)
    d = complex(slider_d.val, slider_d_im.val)
    
    # Check if the transformation is valid (ad - bc ≠ 0)
    det = a*d - b*c
    if abs(det) < 1e-10:
        plt.title(f'Invalid Möbius Transformation: det(ad - bc) ≈ 0')
        return
    
    Z_new = mobius_transform(Z, a, b, c, d)
    img.set_data(complex_to_rgb(Z_new))
    
    # Format complex numbers for display
    def format_complex(z):
        if z.imag == 0:
            return f"{z.real:.2f}"
        elif z.real == 0:
            return f"{z.imag:.2f}i"
        elif z.imag < 0:
            return f"{z.real:.2f} - {abs(z.imag):.2f}i"
        else:
            return f"{z.real:.2f} + {z.imag:.2f}i"
    
    a_str = format_complex(a)
    b_str = format_complex(b)
    c_str = format_complex(c)
    d_str = format_complex(d)
    
    plt.title(f'Möbius Transformation: ({a_str}z + {b_str}) / ({c_str}z + {d_str})')
    fig.canvas.draw_idle()

# Register the complex update function with each slider
slider_a.on_changed(update_complex)
slider_b.on_changed(update_complex)
slider_c.on_changed(update_complex)
slider_d.on_changed(update_complex)
slider_a_im.on_changed(update_complex)
slider_b_im.on_changed(update_complex)
slider_c_im.on_changed(update_complex)
slider_d_im.on_changed(update_complex)

# Add grid and labels
ax.grid(False)
ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')

# Call update_complex once to initialize with complex parameters
update_complex(None)

plt.show()
