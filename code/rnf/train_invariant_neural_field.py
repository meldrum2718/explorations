import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from tqdm import tqdm
import argparse

class NeuralField(nn.Module):
    def __init__(self, d_in, d_out=1, hidden_dim=256, n_layers=3, n_freqs=10):
        super().__init__()
        self.n_freqs = n_freqs
        self.freqs = 2. ** torch.arange(n_freqs, dtype=torch.float32)
        
        pos_dim = d_in * n_freqs * 2
        layers = [nn.Linear(pos_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, d_out))
        self.net = nn.Sequential(*layers)
    
    def positional_encoding(self, x):
        freqs = self.freqs.to(x.device)
        x_freq = x.unsqueeze(-1) * freqs
        return torch.cat([torch.sin(x_freq), torch.cos(x_freq)], dim=-1).flatten(-2)
    
    def forward(self, x):
        shape = x.shape[:-1]
        flat_x = x.reshape(-1, x.shape[-1])
        encoded = self.positional_encoding(flat_x)
        output = self.net(encoded)
        return output.reshape(*shape, -1)

def sample_unit_ball(n_samples, d, device='cpu'):
    """Sample points uniformly from unit ball interior"""
    x = torch.randn(n_samples, d, device=device)
    norms = torch.norm(x, dim=-1, keepdim=True)
    x = x / norms
    r = torch.rand(n_samples, 1, device=device) ** (1/d)
    return x * r

class Transformation:
    """Rotation + scaling transformation in 2D"""
    def __init__(self, angle=0.5, scale_range=(0.8, 1.2)):
        self.angle = angle
        self.scale_min, self.scale_max = scale_range
    
    def __call__(self, x):
        if x.shape[-1] == 2:
            # Random scaling factor for each batch
            scale = torch.rand(x.shape[0], 1, device=x.device, dtype=x.dtype) * (self.scale_max - self.scale_min) + self.scale_min
            
            # Rotation matrix
            cos_a = torch.cos(torch.tensor(self.angle, device=x.device, dtype=x.dtype))
            sin_a = torch.sin(torch.tensor(self.angle, device=x.device, dtype=x.dtype))
            R = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], device=x.device, dtype=x.dtype)
            
            # Apply scaling then rotation
            x_scaled = x * scale
            x_transformed = x_scaled @ R.T
            
            return x_transformed
        else:
            return x

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

def plot_losses(train_losses, test_losses):
    """Plot training and test losses"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(train_losses, label='Training Loss', alpha=0.7)
    ax1.set_yscale('log')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    test_steps = np.arange(0, len(train_losses), 10)[:len(test_losses)]
    ax2.plot(test_steps, test_losses, label='Test Loss', color='orange')
    ax2.set_yscale('log')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss')
    ax2.set_title('Test Loss (Invariance on Random Points)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig

def train_invariant_field(nf, transform, n_epochs=1000, batch_size=1024, lr=1e-3, d=2, n_frames=50):
    """Train neural field to be invariant under transformation"""
    device = next(nf.parameters()).device
    optimizer = torch.optim.Adam(nf.parameters(), lr=lr)
    
    train_losses = []
    test_losses = []
    frames = []
    
    steps_per_frame = max(1, n_epochs // n_frames)
    
    for epoch in tqdm(range(n_epochs + 1)):
        # Generate training batch
        x = sample_unit_ball(batch_size, d, device)
        x_transformed = transform(x)
        
        # Forward pass
        f_x = nf(x)
        f_tx = nf(x_transformed)
        
        # Invariance loss
        loss = F.mse_loss(f_x, f_tx)
        train_losses.append(loss.item())
        
        # Optimization step
        if epoch < n_epochs:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Test loss on separate batch
        if epoch % 10 == 0:
            with torch.no_grad():
                x_test = sample_unit_ball(batch_size, d, device)
                x_test_transformed = transform(x_test)
                f_test = nf(x_test)
                f_test_tx = nf(x_test_transformed)
                test_loss = F.mse_loss(f_test, f_test_tx)
                test_losses.append(test_loss.item())
        
        # Save frame for animation
        if epoch % steps_per_frame == 0:
            frame = generate_2d_visualization(nf, resolution=128)
            frames.append(frame)
            print(f"Epoch {epoch}, Train Loss: {loss.item():.6f}")
    
    return train_losses, test_losses, frames

def parse_args():
    parser = argparse.ArgumentParser(description='Train invariant neural fields')
    
    # Architecture parameters
    parser.add_argument('--d_in', type=int, default=2, help='Input dimension')
    parser.add_argument('--d_out', type=int, default=1, help='Output dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of hidden layers')
    parser.add_argument('--n_freqs', type=int, default=10, help='Number of positional encoding frequencies')
    
    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    # Transformation parameters
    parser.add_argument('--angle', type=float, default=0.05, help='Rotation angle in radians')
    parser.add_argument('--scale_min', type=float, default=0.9, help='Minimum scale factor')
    parser.add_argument('--scale_max', type=float, default=0.9, help='Maximum scale factor')
    
    # Visualization parameters
    parser.add_argument('--n_frames', type=int, default=20, help='Number of animation frames')
    parser.add_argument('--resolution', type=int, default=256, help='Final visualization resolution')
    parser.add_argument('--save_gif', type=str, default=None, help='Path to save animation GIF')
    
    # Device
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda', 'mps'], help='Device to use')
    
    return parser.parse_args()

def get_device(device_arg):
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device_arg)
# Usage
if __name__ == "__main__":
    args = parse_args()
    device = get_device(args.device)
    
    print(f"Training on device: {device}")
    print(f"Architecture: {args.d_in}D → {args.hidden_dim}×{args.n_layers} → {args.d_out}D")
    print(f"Training: {args.n_epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    print(f"Transformation: rotation={args.angle:.2f}rad, scale=[{args.scale_min}, {args.scale_max}]")
    
    # Create neural field with specified architecture
    nf = NeuralField(
        d_in=args.d_in, 
        d_out=args.d_out,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_freqs=args.n_freqs
    ).to(device)
    
    # Create transformation with specified parameters
    transform = Transformation(
        angle=args.angle, 
        scale_range=(args.scale_min, args.scale_max)
    )
    
    # Train with specified parameters
    train_losses, test_losses, frames = train_invariant_field(
        nf=nf,
        transform=transform,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d=args.d_in,
        n_frames=args.n_frames
    )
    
    # Plot losses
    loss_fig = plot_losses(train_losses, test_losses)
    plt.show()
    
    # Create and show animation
    ani = animate_frames(frames, save_path=args.save_gif)
    plt.show()
    
    # Final visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original field
    final_frame = generate_2d_visualization(nf, resolution=args.resolution)
    
    im1 = ax1.imshow(final_frame, extent=[-1.5, 1.5, -1.5, 1.5], cmap='viridis', origin='lower')
    circle1 = plt.Circle((0, 0), 1, fill=False, color='white', linewidth=2)
    ax1.add_patch(circle1)
    ax1.set_title('Learned Invariant Field (Rotation + Scaling)')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1)
    
    # Difference after transformation (should be ~0)
    x_test = sample_unit_ball(1000, args.d_in, device)
    x_test_transformed = transform(x_test)
    with torch.no_grad():
        f_orig = nf(x_test)
        f_trans = nf(x_test_transformed)
        diff = (f_orig - f_trans).norm(dim=-1).cpu().numpy()
    
    ax2.scatter(x_test[:, 0].cpu(), x_test[:, 1].cpu(), c=diff, 
                cmap='Reds', s=1, alpha=0.7)
    circle2 = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax2.add_patch(circle2)
    ax2.set_title('||f(x) - f(T(x))|| for Random Points')
    ax2.set_aspect('equal')
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nFinal train loss: {train_losses[-1]:.6f}")
    print(f"Final test loss: {test_losses[-1]:.6f}")
    if args.save_gif:
        print(f"Animation saved to: {args.save_gif}")
