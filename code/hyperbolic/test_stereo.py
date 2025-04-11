import torch

from projections import *

def process_real_number(x_value: float):
    """
    Process a single real number through projection functions:
    1. First path: central_projection_to_hemisphere -> stereographic_projection_to_flat
    2. Second path: stereographic_projection_to_sphere -> central_projection_to_flat
    
    Parameters:
    -----------
    x_value : float
        A real number to process
        
    Returns:
    --------
    None (prints results)
    """
    print(f"Processing input value: {x_value}")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a 1D tensor with one element (need shape [1,1] for batch processing)
    x_tensor = torch.tensor([[x_value]], dtype=torch.float32, device=device)
    print(f"Input tensor: {x_tensor}")
    
    # Initialize projection parameters (working in 1D + 1D space)
    n = 1  # dimension of flat space
    c = torch.zeros(n+1, device=device)
    c[-1] = 1.0  # Center at [0,1]
    r = 1.0      # Unit radius
    
    print("\n--- Path 1: central_projection_to_hemisphere -> stereographic_projection_to_flat ---")
    try:
        # Step 1: Central projection to hemisphere
        hemisphere_points = central_projection_to_hemisphere(x_tensor, c, r)
        print(f"Hemisphere points: {hemisphere_points}")
        
        # Step 2: Stereographic projection back to flat space
        ball_points = stereographic_projection_to_flat(hemisphere_points, c, r)
        print(f"Ball points (result): {ball_points}")
    except Exception as e:
        print(f"Error in Path 1: {e}")
    
    print("\n--- Path 2: stereographic_projection_to_sphere -> central_projection_to_flat ---")
    try:
        # Step 1: Stereographic projection to sphere
        sphere_points = stereographic_projection_to_sphere(x_tensor, c, r)
        print(f"Sphere points: {sphere_points}")
        
        # Step 2: Central projection back to flat space
        flat_points = central_projection_to_flat(sphere_points, c)
        print(f"Flat points (result): {flat_points}")
    except Exception as e:
        print(f"Error in Path 2: {e}")

# Example usage
if __name__ == "__main__":
    # Process a few different values
    for value in [-2.0, -0.5, 0.0, 0.5, 2.0, 10000]:
        process_real_number(value)
        print("\n" + "="*70 + "\n")
