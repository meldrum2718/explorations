import torch
import torch.nn.functional as F

def flat_to_ball(x, a=1.0, b=0.0):
    """Maps x to tanh(||ax+b||)(ax+b)/||ax+b||"""
    ax_b = a * x + b
    norm = torch.norm(ax_b, dim=-1, keepdim=True)
    # Handle zero norm case
    norm = torch.clamp(norm, min=1e-8)
    unit_vec = ax_b / norm
    return torch.tanh(norm) * unit_vec

def ball_to_flat(y, a=1.0, b=0.0):
    """Inverse of flat_to_ball"""
    norm_y = torch.norm(y, dim=-1, keepdim=True)
    
    # Handle zero norm case - when y=0, the original point was -b/a
    zero_mask = norm_y.squeeze(-1) < 1e-8
    
    # For non-zero points
    norm_y_safe = torch.clamp(norm_y, min=1e-8)
    unit_vec = y / norm_y_safe
    
    # For norm_y > 1, map to 1/norm_y
    norm_y_mapped = torch.where(norm_y > 1.0, 1.0 / norm_y, norm_y)
    
    # Inverse of tanh: atanh(r) where r is the mapped norm
    # Clamp to avoid numerical issues
    norm_y_clamped = torch.clamp(norm_y_mapped, max=1.0 - 1e-6)
    inverse_tanh = torch.atanh(norm_y_clamped)
    
    ax_b = inverse_tanh * unit_vec
    result = (ax_b - b) / a
    
    # Handle zero case: when input was -b/a, output should be -b/a
    if zero_mask.any():
        zero_result = torch.full_like(result[0:1], -b/a)
        result[zero_mask] = zero_result.expand_as(result[zero_mask])
    
    return result

def test_mappings():
    """Interactive visual test with sliders for parameters"""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    
    # Create meshgrid
    res = 128
    x = torch.linspace(-3, 3, res)
    y = torch.linspace(-3, 3, res)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    meshgrid = torch.stack([xx, yy], dim=-1)  # Shape: (res, res, 2)
    
    # Initial parameters
    init_a, init_b = 1.5, 0.2
    init_freq1, init_freq2 = 2.0, 3.0
    init_phase1, init_phase2 = 0.5, 1.0
    
    def periodic_func(coords, freq1, freq2, phase1, phase2):
        """sin(freq1*x+phase1) + sin(freq2*y+phase2)"""
        x, y = coords[..., 0], coords[..., 1]
        return torch.sin(freq1 * x + phase1) + torch.sin(freq2 * y + phase2)
    
    def update_plots(a, b, freq1, freq2, phase1, phase2):
        # 1. Original periodic function on regular meshgrid
        f_orig = periodic_func(meshgrid, freq1, freq2, phase1, phase2)
        
        # 2. Periodic function on ball_to_flat(meshgrid)
        meshgrid_b2f = ball_to_flat(meshgrid, a, b)
        f_b2f = periodic_func(meshgrid_b2f, freq1, freq2, phase1, phase2)
        
        # 3. Periodic function on flat_to_ball(meshgrid)
        meshgrid_f2b = flat_to_ball(meshgrid, a, b)
        f_f2b = periodic_func(meshgrid_f2b, freq1, freq2, phase1, phase2)
        
        # 4. Periodic function on f2b(b2f(meshgrid))
        meshgrid_f2b_b2f = flat_to_ball(ball_to_flat(meshgrid, a, b), a, b)
        f_f2b_b2f = periodic_func(meshgrid_f2b_b2f, freq1, freq2, phase1, phase2)
        
        # 5. Periodic function on b2f(f2b(meshgrid))
        meshgrid_b2f_f2b = ball_to_flat(flat_to_ball(meshgrid, a, b), a, b)
        f_b2f_f2b = periodic_func(meshgrid_b2f_f2b, freq1, freq2, phase1, phase2)
        
        return [f_orig, f_b2f, f_f2b, f_f2b_b2f, f_b2f_f2b]
    
    # Create figure and axes
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    plt.subplots_adjust(bottom=0.35)
    
    titles = ['Original', 'b2f(meshgrid)', 'f2b(meshgrid)', 'f2b(b2f(meshgrid))', 'b2f(f2b(meshgrid))']
    
    # Initial plot
    functions = update_plots(init_a, init_b, init_freq1, init_freq2, init_phase1, init_phase2)
    ims = []
    
    for i, (ax, title, func) in enumerate(zip(axes, titles, functions)):
        im = ax.imshow(func.numpy(), extent=[-3, 3, -3, 3], origin='lower', cmap='viridis', vmin=-2, vmax=2)
        ax.set_title(title)
        ax.set_aspect('equal')
        ims.append(im)
    
    # Create sliders
    slider_height = 0.03
    slider_width = 0.15
    
    ax_a = plt.axes([0.1, 0.25, slider_width, slider_height])
    ax_b = plt.axes([0.3, 0.25, slider_width, slider_height])
    ax_freq1 = plt.axes([0.5, 0.25, slider_width, slider_height])
    ax_freq2 = plt.axes([0.7, 0.25, slider_width, slider_height])
    ax_phase1 = plt.axes([0.1, 0.15, slider_width, slider_height])
    ax_phase2 = plt.axes([0.3, 0.15, slider_width, slider_height])
    
    slider_a = Slider(ax_a, 'a', 0.1, 5.0, valinit=init_a)
    slider_b = Slider(ax_b, 'b', -2.0, 2.0, valinit=init_b)
    slider_freq1 = Slider(ax_freq1, 'freq1', 0.1, 100.0, valinit=init_freq1)
    slider_freq2 = Slider(ax_freq2, 'freq2', 0.1, 100.0, valinit=init_freq2)
    slider_phase1 = Slider(ax_phase1, 'phase1', 0.0, 6.28, valinit=init_phase1)
    slider_phase2 = Slider(ax_phase2, 'phase2', 0.0, 6.28, valinit=init_phase2)
    
    def update(val):
        a = slider_a.val
        b = slider_b.val
        freq1 = slider_freq1.val
        freq2 = slider_freq2.val
        phase1 = slider_phase1.val
        phase2 = slider_phase2.val
        
        functions = update_plots(a, b, freq1, freq2, phase1, phase2)
        
        for im, func in zip(ims, functions):
            im.set_array(func.numpy())
        
        fig.canvas.draw()
    
    # Connect sliders to update function
    slider_a.on_changed(update)
    slider_b.on_changed(update)
    slider_freq1.on_changed(update)
    slider_freq2.on_changed(update)
    slider_phase1.on_changed(update)
    slider_phase2.on_changed(update)
    
    plt.show()

if __name__ == "__main__":
    test_mappings()
