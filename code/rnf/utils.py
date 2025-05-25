import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train invariant neural fields')
    
    # Architecture parameters
    parser.add_argument('--d_in', type=int, default=2, help='Input dimension')
    parser.add_argument('--d_out', type=int, default=None, help='Output dimension (default: same as d_in for equivariance, 1 for invariance)')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of hidden layers')
    parser.add_argument('--n_freqs', type=int, default=10, help='Number of positional encoding frequencies')
    
    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--equivariant', action='store_true', help='Train for equivariance instead of invariance')
    parser.add_argument('--n_steps_per_sample', type=int, default=1, help='Number of optimization steps per sample/transformation')
    
    # Point sampling parameters
    parser.add_argument('--use_mesh', action='store_true', help='Use mesh grid instead of random sampling')
    parser.add_argument('--mesh_resolution', type=int, default=64, help='Resolution for mesh grid (per dimension)')
    parser.add_argument('--mesh_domain', type=float, nargs=2, default=[-1.5, 1.5], help='Domain range for mesh [min, max]')
    parser.add_argument('--filter_unit_ball', action='store_true', help='Filter mesh points to unit ball')
    
    # Transformation parameters
    parser.add_argument('--angle', type=float, default=0.5, help='Rotation angle in radians')
    parser.add_argument('--scale_min', type=float, default=0.7, help='Minimum scale factor')
    parser.add_argument('--scale_max', type=float, default=1.3, help='Maximum scale factor')
    parser.add_argument('--transform_type', type=str, default='rotation_scaling', 
                       choices=['rotation', 'scaling', 'rotation_scaling'],
                       help='Type of transformation to use')
    
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


def print_config(args, device):
    """Print training configuration"""
    print(f"Training on device: {device}")
    print(f"Architecture: {args.d_in}D → {args.hidden_dim}×{args.n_layers} → {args.d_out}D")
    print(f"Training: {args.n_epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    if args.n_steps_per_sample > 1:
        print(f"  - {args.n_steps_per_sample} optimization steps per sample (higher fidelity)")
    print(f"Mode: {'Equivariant' if args.equivariant else 'Invariant'}")
    
    # Point sampling info
    if args.use_mesh:
        print(f"Point sampling: Mesh grid ({args.mesh_resolution}^{args.d_in} = {args.mesh_resolution**args.d_in} points)")
        print(f"  - Domain: {args.mesh_domain}")
        if args.filter_unit_ball:
            print(f"  - Filtered to unit ball")
    else:
        print(f"Point sampling: Random from unit ball ({args.batch_size} per batch)")
    
    print(f"Transformation: {args.transform_type}")
    if 'rotation' in args.transform_type:
        print(f"  - Rotation angle: {args.angle:.2f} rad")
    if 'scaling' in args.transform_type:
        print(f"  - Scale range: [{args.scale_min}, {args.scale_max}]")


def create_transform(args):
    """Create transformation based on args"""
    from transformations import Transformation, RotationTransformation, ScalingTransformation
    
    if args.transform_type == 'rotation':
        return RotationTransformation(angle=args.angle)
    elif args.transform_type == 'scaling':
        return ScalingTransformation(scale_range=(args.scale_min, args.scale_max))
    elif args.transform_type == 'rotation_scaling':
        return Transformation(angle=args.angle, scale_range=(args.scale_min, args.scale_max))
    else:
        raise ValueError(f"Unknown transform type: {args.transform_type}")
