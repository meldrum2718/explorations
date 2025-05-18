import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class TensorWaveSimulation:
    """
    Simulates a wave equation using tensors, working directly with the density field.
    Refactored to separate the dynamics calculation from the step function.
    """
    
    def __init__(self, resolution=100, domain=(-5, 5), dt=0.1, c=0.8, damping=0.005):
        """
        Initialize the tensor-based wave simulation.
        
        Parameters:
        -----------
        resolution : int
            Grid resolution
        domain : tuple
            (min, max) values for coordinates
        dt : float
            Time step size
        c : float
            Wave propagation speed
        damping : float
            Damping coefficient (energy dissipation)
        """
        self.resolution = resolution
        self.domain = domain
        self.dt = dt
        self.c = c
        self.damping = damping
        self.stereo_weight = 0.0  # Default to regular Laplacian (0 = flat, 1 = stereographic)
        self.rotation_matrix = np.eye(3)  # Default to identity rotation
        
        # Create coordinate mesh for visualization
        self.x = np.linspace(domain[0], domain[1], resolution)
        self.y = np.linspace(domain[0], domain[1], resolution)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Initialize wave field with zeros - we'll store two time steps
        # u[0] is the current state, u[1] is the previous state
        self.u = torch.zeros(2, resolution, resolution)
        
        # Initialize with a pulse
        self.reset("center")
        
        # For stereographic projection
        self.r = 1.0  # Radius parameter
        
        # Precompute coordinate grids for the stereographic projection
        self.create_stereo_grids()
    
    def set_radius(self, radius):
        """
        Set the radius parameter for stereographic projection.
        
        Parameters:
        -----------
        radius : float
            Radius parameter for stereographic projection
        """
        self.r = radius
        
        # Update the stereo grids with the new radius
        self.create_stereo_grids()
    
    def create_stereo_grids(self):
        """
        Precompute the coordinate grids for stereographic projection and its inverse.
        """
        # Original grid coordinates
        y, x = torch.meshgrid(
            torch.linspace(self.domain[0], self.domain[1], self.resolution),
            torch.linspace(self.domain[0], self.domain[1], self.resolution),
            indexing='ij'
        )
        
        # Save flat grid
        self.grid_flat = torch.stack([x, y], dim=-1)
        
        # Compute stereographic projection coordinates
        # Map plane to sphere
        r_sq = x**2 + y**2
        denom = r_sq + self.r**2
        
        # Forward projection (plane to sphere)
        x_sphere = 2 * self.r**2 * x / denom
        y_sphere = 2 * self.r**2 * y / denom
        z_sphere = self.r * (r_sq - self.r**2) / denom
        
        self.grid_sphere = torch.stack([x_sphere, y_sphere, z_sphere], dim=-1)
        
        # Keep track of the original indices for each coordinate
        self.indices = torch.stack([
            torch.arange(self.resolution).view(-1, 1).expand(-1, self.resolution),
            torch.arange(self.resolution).view(1, -1).expand(self.resolution, -1)
        ], dim=-1)
    
    def laplacian(self, u):
        """
        Compute the Laplacian of a field using finite differences.
        Uses a 5-point stencil: (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4*u_{i,j})
        
        Parameters:
        -----------
        u : torch.Tensor
            Input tensor of shape [H, W]
            
        Returns:
        --------
        torch.Tensor
            Laplacian of u
        """
        # Pad the tensor to handle boundary conditions
        padded = F.pad(u.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
        
        # Extract neighbors
        center = padded[:, :, 1:-1, 1:-1]
        left = padded[:, :, 1:-1, :-2]
        right = padded[:, :, 1:-1, 2:]
        top = padded[:, :, :-2, 1:-1]
        bottom = padded[:, :, 2:, 1:-1]
        
        # 5-point stencil laplacian
        return (left + right + top + bottom - 4 * center).squeeze()
    
    def apply_stereographic_projection(self, u, radius=None, rotation_matrix=None):
        """
        Apply stereographic projection to a field and return it.
        
        Parameters:
        -----------
        u : torch.Tensor
            Input tensor of shape [H, W]
        radius : float, optional
            Radius parameter for stereographic projection. If None, use self.r
        rotation_matrix : numpy.ndarray, optional
            3x3 rotation matrix to apply to the points on the sphere
            
        Returns:
        --------
        torch.Tensor
            Field after stereographic projection and back-projection
        """
        # Use provided radius or default
        radius = radius if radius is not None else self.r
        
        # Get original grid coordinates
        y, x = torch.meshgrid(
            torch.linspace(self.domain[0], self.domain[1], self.resolution),
            torch.linspace(self.domain[0], self.domain[1], self.resolution),
            indexing='ij'
        )
        
        # Step 1: Map each point (x,y) to a point on the sphere
        r_sq = x**2 + y**2
        denom = r_sq + radius**2
        
        X_sphere = 2 * radius**2 * x / denom
        Y_sphere = 2 * radius**2 * y / denom
        Z_sphere = radius * (r_sq - radius**2) / denom
        
        # Step 2: Apply rotation if provided
        if rotation_matrix is not None:
            # Convert rotation matrix to tensor if it's numpy
            if isinstance(rotation_matrix, np.ndarray):
                rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32)
                
            # Prepare points for rotation
            points = torch.stack((X_sphere, Y_sphere, Z_sphere), dim=-1)  # [H, W, 3]
            points_flat = points.reshape(-1, 3)  # [H*W, 3]
            
            # Apply rotation
            rotated_points = torch.matmul(points_flat, rotation_matrix.T)  # [H*W, 3]
            rotated_points = rotated_points.reshape(self.resolution, self.resolution, 3)  # [H, W, 3]
            
            # Extract rotated coordinates
            X_sphere = rotated_points[:, :, 0]
            Y_sphere = rotated_points[:, :, 1]
            Z_sphere = rotated_points[:, :, 2]
        
        # Step 3: Project back to plane
        # Handle points where Z_rot == 1 (division by zero)
        mask = torch.abs(1 - Z_sphere) > 1e-10
        X_proj = torch.zeros_like(X_sphere)
        Y_proj = torch.zeros_like(Y_sphere)
        
        X_proj[mask] = X_sphere[mask] / (1 - Z_sphere[mask])
        Y_proj[mask] = Y_sphere[mask] / (1 - Z_sphere[mask])
        
        # Step 4: Sample the original field at the projected coordinates
        # Normalize coordinates to [0, resolution-1]
        X_idx = ((X_proj - self.domain[0]) / (self.domain[1] - self.domain[0])) * (self.resolution - 1)
        Y_idx = ((Y_proj - self.domain[0]) / (self.domain[1] - self.domain[0])) * (self.resolution - 1)
        
        # Clamp to valid range
        X_idx = torch.clamp(X_idx, 0, self.resolution - 1)
        Y_idx = torch.clamp(Y_idx, 0, self.resolution - 1)
        
        # Grid sample requires normalized coordinates [-1, 1]
        X_norm = 2.0 * X_idx / (self.resolution - 1) - 1.0
        Y_norm = 2.0 * Y_idx / (self.resolution - 1) - 1.0
        
        # Stack coordinates for grid_sample
        grid = torch.stack([X_norm, Y_norm], dim=-1).unsqueeze(0)
        
        # Sample the field using grid_sample
        u_reshaped = u.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        result = F.grid_sample(u_reshaped, grid, mode='bilinear', padding_mode='reflection', align_corners=True)
        
        return result.squeeze()

    def set_stereo_weight(self, weight):
        """
        Set the weight for blending between the flat wave field and its stereographic projection.
        
        Parameters:
        -----------
        weight : float
            Weight between 0 and 1, where:
            0 = regular flat wave field
            1 = fully stereographically projected wave field
            Values in between blend the two fields
        """
        self.stereo_weight = weight
    
    def compute_acceleration(self, u_curr, u_prev, dt, c, damping, stereo_weight=None, rotation_matrix=None):
        """
        Compute the acceleration term (second time derivative) for the wave equation.
        Now separated from the step function to allow external modifications.
        
        Parameters:
        -----------
        u_curr : torch.Tensor
            Current state of the field
        u_prev : torch.Tensor
            Previous state of the field
        dt : float
            Time step
        c : float
            Wave speed
        damping : float
            Damping coefficient
        stereo_weight : float, optional
            Weight for stereographic projection blending
        rotation_matrix : numpy.ndarray, optional
            Rotation matrix for stereographic projection
            
        Returns:
        --------
        torch.Tensor
            Acceleration term
        """
        # Use provided parameters or defaults
        stereo_weight = stereo_weight if stereo_weight is not None else self.stereo_weight
        rotation_matrix = rotation_matrix if rotation_matrix is not None else self.rotation_matrix
        
        # Apply stereographic projection if needed
        if stereo_weight > 0:
            # Get stereographically projected version of the current field
            u_stereo = self.apply_stereographic_projection(u_curr, rotation_matrix=rotation_matrix)
            
            # Blend the original field with its stereographic projection
            u_blended = (1 - stereo_weight) * u_curr + stereo_weight * u_stereo
        else:
            # No blending needed
            u_blended = u_curr
        
        # Compute Laplacian of the blended field
        lapl = self.laplacian(u_blended)
        
        # Apply normalization to Laplacian to prevent excessive growth
        # Clip large values while preserving sign
        max_lapl_value = 1.0
        lapl_sign = torch.sign(lapl)
        lapl_abs = torch.abs(lapl)
        lapl_normalized = lapl_sign * torch.minimum(lapl_abs, torch.tensor(max_lapl_value))
        
        # Compute the acceleration term from the wave equation
        # a = c²*∇²u - d*v
        # where v = (u_curr - u_prev)/dt is the velocity
        
        # Use a higher coefficient for the Laplacian to make waves propagate faster
        laplacian_factor = c * c
        velocity = (u_blended - u_prev) / dt
        
        # Acceleration = c²*∇²u - d*v
        acceleration = laplacian_factor * lapl_normalized - damping * velocity
        
        return acceleration, u_blended
    
    def step(self, external_acceleration=None):
        """
        Advance simulation by one time step using wave equation.
        Optionally accepts external acceleration term.
        
        Parameters:
        -----------
        external_acceleration : torch.Tensor, optional
            External acceleration to add to the computed acceleration
            
        Returns:
        --------
        numpy.ndarray
            Updated density field
        """
        # Get current and previous state
        u_curr = self.u[0]
        u_prev = self.u[1]
        
        # Compute acceleration
        acceleration, u_blended = self.compute_acceleration(
            u_curr, u_prev, self.dt, self.c, self.damping, 
            self.stereo_weight, self.rotation_matrix
        )
        
        # Add external acceleration if provided
        if external_acceleration is not None:
            acceleration = acceleration + external_acceleration
        
        # Wave equation discretization with explicit scheme
        # u_next = 2*u_curr - u_prev + dt²*acceleration
        dt_sq = self.dt * self.dt
        
        u_next = 2 * u_blended - u_prev + dt_sq * acceleration
        
        # Apply normalization to keep the wave amplitude in check
        # This prevents numerical instability from amplitudes growing too large
        max_amplitude = 2.0
        u_next = torch.clamp(u_next, -max_amplitude, max_amplitude)
        
        # Periodic normalization to prevent long-term drift
        # Every few steps, scale down the wave if its overall energy is too high
        if torch.rand(1).item() < 0.05:  # Randomly apply with 5% chance to avoid regular patterns
            energy = torch.sum(u_next**2)
            max_energy = self.resolution * self.resolution * 0.5  # Reasonable energy threshold
            if energy > max_energy:
                scaling_factor = torch.sqrt(max_energy / energy)
                u_next = u_next * scaling_factor
        
        # Update state history
        self.u[1] = u_curr.clone()  # Previous becomes current
        self.u[0] = u_next.clone()  # Current becomes next
        
        # Convert to numpy for visualization
        return self.u[0].cpu().detach().numpy()
    
    def reset(self, pulse_type="center"):
        """
        Reset the simulation with a new initial pulse.
        
        Parameters:
        -----------
        pulse_type : str
            Type of initial condition ("center", "random", "double", "ripple")
        """
        # Reset the state
        self.u.zero_()
        
        # Create initial pulse based on pulse type
        if pulse_type == "center":
            # Single pulse at center
            self.u[0] = self.gaussian_pulse([0, 0], 0.5)
            
        elif pulse_type == "random":
            # Multiple random pulses
            num_pulses = np.random.randint(3, 7)
            for _ in range(num_pulses):
                x_pos = np.random.uniform(self.domain[0] * 0.7, self.domain[1] * 0.7)
                y_pos = np.random.uniform(self.domain[0] * 0.7, self.domain[1] * 0.7)
                width = np.random.uniform(0.2, 0.8)
                
                self.u[0] += self.gaussian_pulse([x_pos, y_pos], width) * np.random.uniform(0.5, 1.0)
            
        elif pulse_type == "double":
            # Two pulses of opposite sign
            self.u[0] = self.gaussian_pulse([-2, 2], 0.5) - self.gaussian_pulse([2, -2], 0.5)
            
        elif pulse_type == "ripple":
            # Circular ripple pattern
            y, x = torch.meshgrid(
                torch.linspace(self.domain[0], self.domain[1], self.resolution),
                torch.linspace(self.domain[0], self.domain[1], self.resolution),
                indexing='ij'
            )
            r = torch.sqrt(x**2 + y**2)
            k = 3.0  # Wave number
            self.u[0] = torch.sin(k * r) * torch.exp(-r**2 / 8)
        
        # Initialize the second time step with the same value (zero initial velocity)
        self.u[1] = self.u[0].clone()
    
    def gaussian_pulse(self, center, width):
        """
        Create a Gaussian pulse centered at the specified coordinates.
        
        Parameters:
        -----------
        center : list
            [x, y] coordinates of the center
        width : float
            Standard deviation of the Gaussian
            
        Returns:
        --------
        torch.Tensor
            2D Gaussian pulse
        """
        # Create coordinate grid
        y, x = torch.meshgrid(
            torch.linspace(self.domain[0], self.domain[1], self.resolution),
            torch.linspace(self.domain[0], self.domain[1], self.resolution),
            indexing='ij'
        )
        
        # Compute Gaussian
        cx, cy = center
        r_sq = (x - cx)**2 + (y - cy)**2
        return torch.exp(-r_sq / (2 * width**2))
    
    def get_density(self):
        """
        Get the current density field.
        
        Returns:
        --------
        numpy.ndarray
            Current density field
        """
        return self.u[0].cpu().detach().numpy()
    
    def set_rotation_matrix(self, rotation_matrix):
        """
        Set the rotation matrix used for stereographic projection.
        
        Parameters:
        -----------
        rotation_matrix : numpy.ndarray
            3x3 rotation matrix
        """
        self.rotation_matrix = rotation_matrix


def wave_tensor_field(x, y, simulation, mode='density'):
    """
    Sample the wave field at given coordinates for stereographic projection.
    
    Parameters:
    -----------
    x, y : numpy.ndarray
        Coordinates at which to sample the field
    simulation : TensorWaveSimulation
        Simulation object containing the field
    mode : str
        Visualization mode (only 'density' is used now)
        
    Returns:
    --------
    numpy.ndarray
        Field values at the given coordinates
    """
    # Convert input coordinates to grid indices
    domain = simulation.domain
    resolution = simulation.resolution
    
    # Normalize coordinates to [0, resolution-1]
    x_idx = ((x - domain[0]) / (domain[1] - domain[0])) * (resolution - 1)
    y_idx = ((y - domain[0]) / (domain[1] - domain[0])) * (resolution - 1)
    
    # Clamp indices to valid range
    x_idx = np.clip(x_idx, 0, resolution - 1)
    y_idx = np.clip(y_idx, 0, resolution - 1)
    
    # Bilinear interpolation
    x0 = np.floor(x_idx).astype(int)
    y0 = np.floor(y_idx).astype(int)
    x1 = np.minimum(x0 + 1, resolution - 1)
    y1 = np.minimum(y0 + 1, resolution - 1)
    
    wx = x_idx - x0
    wy = y_idx - y0
    
    # Get the current density field
    field = simulation.get_density()
    
    # Interpolate
    result = (
        field[y0, x0] * (1 - wx) * (1 - wy) +
        field[y0, x1] * wx * (1 - wy) +
        field[y1, x0] * (1 - wx) * wy +
        field[y1, x1] * wx * wy
    )
    
    return result
