import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from tqdm import tqdm

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
    """Example: rotation in 2D"""
    def __init__(self, angle=0.5):
        self.angle = angle
    
    def __call__(self, x):
        if x.shape[-1] == 2:
            cos_a = torch.cos(torch.tensor(self.angle, device=x.device, dtype=x.dtype))
            sin_a = torch.sin(torch.tensor(self.angle, device=x.device, dtype=x.dtype))
            R = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], device=x.device, dtype=x.dtype)
            return x @ R.T
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
        values = nf(coords).squeeze(-1)
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

def train_invariant_field(nf, transform, n_steps=1000, batch_size=1024, lr=1e-3, d=2, n_frames=50):
    """Train neural field to be invariant under transformation"""
    device = next(nf.parameters()).device
    optimizer = torch.optim.Adam(nf.parameters(), lr=lr)
    
    train_losses = []
    test_losses = []
    frames = []
    
    steps_per_frame = max(1, n_steps // n_frames)
    
    for step in tqdm(range(n_steps + 1)):
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
        if step < n_steps:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Test loss on separate batch
        if step % 10 == 0:
            with torch.no_grad():
                x_test = sample_unit_ball(batch_size, d, device)
                x_test_transformed = transform(x_test)
                f_test = nf(x_test)
                f_test_tx = nf(x_test_transformed)
                test_loss = F.mse_loss(f_test, f_test_tx)
                test_losses.append(test_loss.item())
        
        # Save frame for animation
        if step % steps_per_frame == 0:
            frame = generate_2d_visualization(nf, resolution=128)
            frames.append(frame)
            print(f"Step {step}, Train Loss: {loss.item():.6f}")
    
    return train_losses, test_losses, frames

# Usage
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2D scalar field invariant under rotation
    nf = NeuralField(d_in=2, d_out=1).to(device)
    transform = Transformation(angle=0.1)  # 0.5 radian rotation
    
    # Train with visualization
    train_losses, test_losses, frames = train_invariant_field(
        nf, transform, n_steps=1000, d=2, n_frames=20
    )
    
    # Plot losses
    loss_fig = plot_losses(train_losses, test_losses)
    plt.show()
    
    # Create and show animation
    ani = animate_frames(frames)
    plt.show()
    
    # Final visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original field
    final_frame = generate_2d_visualization(nf, resolution=256)
    im1 = ax1.imshow(final_frame, extent=[-1.5, 1.5, -1.5, 1.5], cmap='viridis', origin='lower')
    circle1 = plt.Circle((0, 0), 1, fill=False, color='white', linewidth=2)
    ax1.add_patch(circle1)
    ax1.set_title('Learned Invariant Field')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1)
    
    # Difference after transformation (should be ~0)
    x_test = sample_unit_ball(1000, 2, device)
    x_test_rot = transform(x_test)
    with torch.no_grad():
        diff = (nf(x_test) - nf(x_test_rot)).abs().cpu().numpy()
    
    ax2.scatter(x_test[:, 0].cpu(), x_test[:, 1].cpu(), c=diff.flatten(), 
                cmap='Reds', s=1, alpha=0.7)
    circle2 = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax2.add_patch(circle2)
    ax2.set_title('|f(x) - f(T(x))| for Random Points')
    ax2.set_aspect('equal')
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    
    plt.tight_layout()
    plt.show()
