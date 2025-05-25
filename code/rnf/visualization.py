import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def generate_2d_visualization(nf, resolution=128):
    """Generate 2D visualization of the neural field"""
    device = next(nf.parameters()).device
    x = torch.linspace(-1.5, 1.5, resolution, device=device)
    y = torch.linspace(-1.5, 1.5, resolution, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    coords = torch.stack([xx, yy], dim=-1)
    
    with torch.no_grad():
        values = nf(coords)
        # Handle multi-dimensional output by taking the norm
        if values.shape[-1] > 1:
            values = torch.norm(values, dim=-1)
        else:
            values = values.squeeze(-1)
    return values.cpu().numpy()


def animate_frames(frames, save_path=None):
    """Create animation from frames"""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_title('Neural Field Evolution')
    
    circle = plt.Circle((0, 0), 1, fill=False, color='white', linewidth=2)
    ax.add_patch(circle)
    
    im = ax.imshow(frames[0], extent=[-1.5, 1.5, -1.5, 1.5], cmap='viridis', origin='lower')
    plt.colorbar(im, ax=ax)
    
    def update(frame_idx):
        im.set_data(frames[frame_idx])
        im.set_clim(frames[frame_idx].min(), frames[frame_idx].max())
        ax.set_title(f'Neural Field Evolution - Frame {frame_idx}')
        return [im]
    
    ani = FuncAnimation(fig, update, frames=len(frames), interval=200, blit=False)
    if save_path:
        ani.save(save_path, writer='pillow', fps=5)
    return ani


def plot_losses(train_losses, test_losses, equivariant=False):
    """Plot training and test losses"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    mode = "Equivariant" if equivariant else "Invariant"
    
    ax1.plot(train_losses, label=f'Training Loss ({mode})', alpha=0.7)
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training Loss ({mode})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    test_steps = np.arange(0, len(train_losses), 10)[:len(test_losses)]
    ax2.plot(test_steps, test_losses, label=f'Test Loss ({mode})', color='orange')
    ax2.set_yscale('log')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'Test Loss ({mode} on Random Points)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig


def plot_final_results(nf, transform, args, device):
    """Create final visualization plots"""
    from transformations import sample_unit_ball
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original field
    final_frame = generate_2d_visualization(nf, resolution=args.resolution)
    
    im1 = ax1.imshow(final_frame, extent=[-1.5, 1.5, -1.5, 1.5], cmap='viridis', origin='lower')
    circle1 = plt.Circle((0, 0), 1, fill=False, color='white', linewidth=2)
    ax1.add_patch(circle1)
    mode = "Equivariant" if args.equivariant else "Invariant"
    ax1.set_title(f'Learned {mode} Field')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1)
    
    # Difference after transformation
    x_test = sample_unit_ball(1000, args.d_in, device)
    
    with torch.no_grad():
        if args.equivariant:
            # For equivariance: compare f(T(x)) with T(f(x))
            x_test_transformed, R, scale = transform.transform_points(x_test)
            f_orig = nf(x_test)
            f_trans = nf(x_test_transformed)
            f_orig_transformed = transform.transform_vectors(f_orig, R, scale)
            diff = (f_trans - f_orig_transformed).norm(dim=-1).cpu().numpy()
            title = 'f(T(x)) - T(f(x))'
        else:
            # For invariance: compare f(T(x)) with f(x)
            x_test_transformed = transform(x_test)
            f_orig = nf(x_test)
            f_trans = nf(x_test_transformed)
            diff = (f_orig - f_trans).norm(dim=-1).cpu().numpy()
            title = 'f(x) - f(T(x))'
    
    ax2.scatter(x_test[:, 0].cpu(), x_test[:, 1].cpu(), c=diff, 
                cmap='Reds', s=1, alpha=0.7)
    circle2 = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax2.add_patch(circle2)
    ax2.set_title(f'||{title}|| for Random Points')
    ax2.set_aspect('equal')
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    
    plt.tight_layout()
    return fig
