import torch
import numpy as np
import matplotlib.pyplot as plt
from projections import (
    ball_to_flat, 
    flat_to_ball,
    stereographic_projection_to_sphere,
    stereographic_projection_to_flat,
    central_projection_to_hemisphere,
    central_projection_to_flat
)

def generate_ball_points(num_points=1000, radius=0.99, random_seed=42):
    """Generate random points uniformly distributed in a ball"""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Generate random directions (uniformly on the sphere)
    directions = torch.randn(num_points, 2)
    directions = directions / torch.norm(directions, dim=1, keepdim=True)
    
    # Scale by random radii (ensuring uniform distribution in the ball)
    # For 2D, we need to use sqrt(r) to get uniform distribution in the disk
    radii = torch.rand(num_points, 1).sqrt() * radius
    ball_points = directions * radii
    
    return ball_points, radii

def generate_flat_points(num_points=1000, range_limit=2.0, random_seed=42):
    """Generate random points in flat space"""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Generate points uniformly in a square
    flat_points = torch.rand(num_points, 2) * 2 * range_limit - range_limit
    
    # Calculate distance from origin for coloring
    distances = torch.norm(flat_points, dim=1, keepdim=True)
    
    return flat_points, distances

def test_ball_flat_inverse(num_points=1000, radius=0.99, random_seed=42):
    """Test if ball_to_flat and flat_to_ball are inverse mappings"""
    print("\n===== Testing ball_to_flat and flat_to_ball as inverses =====")
    
    # Generate points in the ball
    ball_points_original, radii = generate_ball_points(num_points, radius, random_seed)
    
    # Apply ball_to_flat mapping
    flat_points = ball_to_flat(ball_points_original)
    
    # Apply flat_to_ball to go back to the ball
    ball_points_reconstructed = flat_to_ball(flat_points)
    
    # Calculate error
    error = torch.norm(ball_points_original - ball_points_reconstructed, dim=1)
    max_error = error.max().item()
    mean_error = error.mean().item()
    
    print(f"Maximum Error: {max_error:.10f}")
    print(f"Mean Error: {mean_error:.10f}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.scatter(ball_points_original[:, 0].numpy(), 
                ball_points_original[:, 1].numpy(), 
                c=radii.squeeze().numpy(), cmap='viridis', s=10)
    plt.title('Original Points in Ball')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    
    plt.subplot(132)
    plt.scatter(flat_points[:, 0].numpy(), 
                flat_points[:, 1].numpy(), 
                c=radii.squeeze().numpy(), cmap='viridis', s=10)
    plt.title('After ball_to_flat')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    
    plt.subplot(133)
    plt.scatter(ball_points_reconstructed[:, 0].numpy(), 
                ball_points_reconstructed[:, 1].numpy(), 
                c=radii.squeeze().numpy(), cmap='viridis', s=10)
    plt.title('After flat_to_ball (Reconstructed)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return max_error, mean_error

def test_stereographic_inverse(num_points=1000, radius=1.0, random_seed=42):
    """Test if stereographic projections are inverse mappings"""
    print("\n===== Testing stereographic projections as inverses =====")
    
    # Generate points in flat space
    flat_points_original, distances = generate_flat_points(num_points, 2.0, random_seed)
    
    # Set up projection parameters
    n = flat_points_original.shape[1]
    c = torch.zeros(n+1)
    c[-1] = 1.0  # Center at [0, 0, 1]
    
    # Apply stereographic projection to sphere
    sphere_points = stereographic_projection_to_sphere(flat_points_original, c, radius)
    
    # Apply stereographic projection back to flat
    flat_points_reconstructed = stereographic_projection_to_flat(sphere_points, c, radius)
    
    # Calculate error
    error = torch.norm(flat_points_original - flat_points_reconstructed, dim=1)
    max_error = error.max().item()
    mean_error = error.mean().item()
    
    print(f"Maximum Error: {max_error:.10f}")
    print(f"Mean Error: {mean_error:.10f}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.scatter(flat_points_original[:, 0].numpy(), 
                flat_points_original[:, 1].numpy(), 
                c=distances.squeeze().numpy(), cmap='viridis', s=10)
    plt.title('Original Points in Flat Space')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    
    # For sphere points, we can project to 2D for visualization
    plt.subplot(132)
    # Normalize sphere points to the unit sphere for better visualization
    normalized_sphere = sphere_points / torch.norm(sphere_points, dim=1, keepdim=True)
    plt.scatter(normalized_sphere[:, 0].numpy(), 
                normalized_sphere[:, 1].numpy(), 
                c=distances.squeeze().numpy(), cmap='viridis', s=10)
    plt.title('On Sphere (Projected to 2D)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    
    plt.subplot(133)
    plt.scatter(flat_points_reconstructed[:, 0].numpy(), 
                flat_points_reconstructed[:, 1].numpy(), 
                c=distances.squeeze().numpy(), cmap='viridis', s=10)
    plt.title('After Reconstruction')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return max_error, mean_error

def test_central_inverse(num_points=1000, radius=1.0, random_seed=42):
    """Test if central projections are inverse mappings"""
    print("\n===== Testing central projections as inverses =====")
    
    # Generate points in flat space
    flat_points_original, distances = generate_flat_points(num_points, 2.0, random_seed)
    
    # Set up projection parameters
    n = flat_points_original.shape[1]
    c = torch.zeros(n+1)
    c[-1] = 1.0  # Center at [0, 0, 1]
    
    # Apply central projection to hemisphere
    hemisphere_points = central_projection_to_hemisphere(flat_points_original, c, radius)
    
    # Apply central projection back to flat
    flat_points_reconstructed = central_projection_to_flat(hemisphere_points, c)
    
    # Calculate error
    error = torch.norm(flat_points_original - flat_points_reconstructed, dim=1)
    max_error = error.max().item()
    mean_error = error.mean().item()
    
    print(f"Maximum Error: {max_error:.10f}")
    print(f"Mean Error: {mean_error:.10f}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.scatter(flat_points_original[:, 0].numpy(), 
                flat_points_original[:, 1].numpy(), 
                c=distances.squeeze().numpy(), cmap='viridis', s=10)
    plt.title('Original Points in Flat Space')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    
    # For hemisphere points, we can project to 2D for visualization
    plt.subplot(132)
    # Normalize hemisphere points for better visualization
    normalized_hemisphere = hemisphere_points / torch.norm(hemisphere_points, dim=1, keepdim=True)
    plt.scatter(normalized_hemisphere[:, 0].numpy(), 
                normalized_hemisphere[:, 1].numpy(), 
                c=distances.squeeze().numpy(), cmap='viridis', s=10)
    plt.title('On Hemisphere (Projected to 2D)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    
    plt.subplot(133)
    plt.scatter(flat_points_reconstructed[:, 0].numpy(), 
                flat_points_reconstructed[:, 1].numpy(), 
                c=distances.squeeze().numpy(), cmap='viridis', s=10)
    plt.title('After Reconstruction')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return max_error, mean_error

def test_all_projections():
    """Run all projection tests"""
    # Test ball_to_flat and flat_to_ball
    ball_flat_max_err, ball_flat_mean_err = test_ball_flat_inverse()
    
    # Test stereographic projections
    stereo_max_err, stereo_mean_err = test_stereographic_inverse()
    
    # Test central projections
    central_max_err, central_mean_err = test_central_inverse()
    
    # Summary
    print("\n===== Summary of Results =====")
    print(f"Ball-Flat Mapping: Max Error = {ball_flat_max_err:.10f}, Mean Error = {ball_flat_mean_err:.10f}")
    print(f"Stereographic: Max Error = {stereo_max_err:.10f}, Mean Error = {stereo_mean_err:.10f}")
    print(f"Central: Max Error = {central_max_err:.10f}, Mean Error = {central_mean_err:.10f}")

# Run all tests
if __name__ == "__main__":
    test_all_projections()
    
    # Additional focused tests
    print("\n===== Additional Tests =====")
    print("Testing with points near the boundary...")
    test_ball_flat_inverse(num_points=500, radius=0.999)
    test_stereographic_inverse(num_points=500, radius=0.999)
    test_central_inverse(num_points=500, radius=0.999)
