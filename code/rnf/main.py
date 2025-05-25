#!/usr/bin/env python3
"""
Main script for training invariant neural fields.

Usage:
    python main.py --n_epochs 2000 --hidden_dim 512 --transform_type rotation --angle 1.0
"""

import matplotlib.pyplot as plt

from neural_field import NeuralField
from transformations import sample_unit_ball
from training import train_invariant_field
from visualization import plot_losses, animate_frames, plot_final_results
from utils import parse_args, get_device, print_config, create_transform


def main():
    # Parse arguments
    args = parse_args()
    device = get_device(args.device)
    
    # Set default d_out based on mode
    if args.d_out is None:
        args.d_out = args.d_in if args.equivariant else 1
    
    # Print configuration
    print_config(args, device)
    
    # Create neural field with specified architecture
    nf = NeuralField(
        d_in=args.d_in, 
        d_out=args.d_out,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_freqs=args.n_freqs
    ).to(device)
    
    # Create transformation
    transform = create_transform(args)
    
    # Train with specified parameters
    train_losses, test_losses, frames = train_invariant_field(
        nf=nf,
        transform=transform,
        args=args,
        n_frames=args.n_frames
    )
    
    print(f"\nFinal train loss: {train_losses[-1]:.6f}")
    print(f"Final test loss: {test_losses[-1]:.6f}")
    
    # Plot losses
    loss_fig = plot_losses(train_losses, test_losses, equivariant=args.equivariant)
    plt.show()
    
    # Create and show animation
    ani = animate_frames(frames, save_path=args.save_gif)
    if args.save_gif:
        print(f"Animation saved to: {args.save_gif}")
    plt.show()
    
    # Final visualization
    final_fig = plot_final_results(nf, transform, args, device)
    plt.show()


if __name__ == "__main__":
    main()
