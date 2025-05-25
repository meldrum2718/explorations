import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformations import sample_unit_ball, generate_mesh_grid, filter_points_in_unit_ball
from visualization import generate_2d_visualization


def get_training_points(args, device, epoch=None):
    """Get training points based on configuration"""
    if args.use_mesh:
        # Generate mesh grid points
        mesh_points = generate_mesh_grid(
            resolution=args.mesh_resolution,
            d=args.d_in,
            domain_range=tuple(args.mesh_domain),
            device=device
        )
        
        if args.filter_unit_ball:
            mesh_points = filter_points_in_unit_ball(mesh_points)
        
        # If using batches, randomly sample from mesh
        if len(mesh_points) > args.batch_size:
            indices = torch.randperm(len(mesh_points))[:args.batch_size]
            return mesh_points[indices]
        else:
            return mesh_points
    else:
        # Random sampling from unit ball
        return sample_unit_ball(args.batch_size, args.d_in, device)


def train_invariant_field(nf, transform, args, n_frames=50):
    """Train neural field to be invariant or equivariant under transformation"""
    device = next(nf.parameters()).device
    optimizer = torch.optim.Adam(nf.parameters(), lr=args.lr)
    
    train_losses = []
    test_losses = []
    frames = []
    
    steps_per_frame = max(1, args.n_epochs // n_frames)
    
    # Pre-generate mesh if using mesh-based training
    if args.use_mesh:
        print(f"Generating mesh grid with {args.mesh_resolution}^{args.d_in} points...")
        all_mesh_points = generate_mesh_grid(
            resolution=args.mesh_resolution,
            d=args.d_in,
            domain_range=tuple(args.mesh_domain),
            device=device
        )
        if args.filter_unit_ball:
            all_mesh_points = filter_points_in_unit_ball(all_mesh_points)
        print(f"Using {len(all_mesh_points)} training points")
    
    step_counter = 0
    for epoch in tqdm(range(args.n_epochs + 1)):
        # Get training points
        if args.use_mesh and epoch > 0:  # Skip first epoch for frame generation
            # Use all mesh points or random subset
            if len(all_mesh_points) <= args.batch_size:
                x = all_mesh_points
            else:
                indices = torch.randperm(len(all_mesh_points))[:args.batch_size]
                x = all_mesh_points[indices]
        else:
            x = get_training_points(args, device, epoch)
        
        # Sample transformation parameters once per epoch
        if args.equivariant:
            x_transformed, R, scale = transform.transform_points(x)
        else:
            x_transformed = transform(x)
        
        # Multiple optimization steps per sample for higher fidelity
        epoch_loss = 0.0
        for step in range(args.n_steps_per_sample):
            if args.equivariant:
                # Equivariance: f(T(x)) = T(f(x))
                # Use the same transformation for all steps
                f_x = nf(x)
                f_tx = nf(x_transformed)
                
                # Transform the output of f(x) using the same transformation
                f_x_transformed = transform.transform_vectors(f_x, R, scale)
                
                # Equivariance loss: f(T(x)) should equal T(f(x))
                loss = F.mse_loss(f_tx, f_x_transformed)
            else:
                # Invariance: f(T(x)) = f(x)
                f_x = nf(x)
                f_tx = nf(x_transformed)
                
                # Invariance loss
                loss = F.mse_loss(f_x, f_tx)
            
            epoch_loss += loss.item()
            
            # Optimization step
            if epoch < args.n_epochs:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step_counter += 1
        
        # Record average loss for this epoch
        avg_epoch_loss = epoch_loss / args.n_steps_per_sample
        train_losses.append(avg_epoch_loss)
        
        # Test loss on separate batch (always use random sampling for test)
        if epoch % 10 == 0:
            with torch.no_grad():
                x_test = sample_unit_ball(args.batch_size, args.d_in, device)
                
                if args.equivariant:
                    x_test_transformed, R_test, scale_test = transform.transform_points(x_test)
                    f_test = nf(x_test)
                    f_test_tx = nf(x_test_transformed)
                    f_test_transformed = transform.transform_vectors(f_test, R_test, scale_test)
                    test_loss = F.mse_loss(f_test_tx, f_test_transformed)
                else:
                    x_test_transformed = transform(x_test)
                    f_test = nf(x_test)
                    f_test_tx = nf(x_test_transformed)
                    test_loss = F.mse_loss(f_test, f_test_tx)
                
                test_losses.append(test_loss.item())
        
        # Save frame for animation
        if epoch % steps_per_frame == 0:
            frame = generate_2d_visualization(nf, resolution=128)
            frames.append(frame)
            mode = "Equivariant" if args.equivariant else "Invariant"
            sampling = "Mesh" if args.use_mesh else "Random"
            steps_info = f" (steps/sample: {args.n_steps_per_sample})" if args.n_steps_per_sample > 1 else ""
            print(f"Epoch {epoch}, {mode} Loss: {avg_epoch_loss:.6f} ({sampling}{steps_info})")
    
    total_steps = step_counter
    print(f"Total optimization steps: {total_steps}")
    
    return train_losses, test_losses, frames
