import torch
import torch.nn as nn

def fft_conv_nd(x, weight, bias=None, debug=False):
    """Fast vectorized N-dimensional convolution using FFT"""
    batch_size, in_ch, *spatial_dims = x.shape
    out_ch, in_ch_w, *kernel_dims = weight.shape
    ndim = len(spatial_dims)
    
    if debug:
        print(f"FFT Conv Debug:")
        print(f"  Input shape: {x.shape}")
        print(f"  Weight shape: {weight.shape}")
        print(f"  Spatial dims: {spatial_dims}")
        print(f"  Kernel dims: {kernel_dims}")
    
    # Flip kernel for convolution (FFT gives correlation, we want convolution)
    # Flip all spatial dimensions
    flip_dims = list(range(2, 2 + ndim))
    weight_flipped = torch.flip(weight, dims=flip_dims)
    
    # Compute output size for valid convolution
    valid_sizes = [s - k + 1 for s, k in zip(spatial_dims, kernel_dims)]
    
    # Compute size needed to avoid circular convolution
    conv_sizes = [s + k - 1 for s, k in zip(spatial_dims, kernel_dims)]
    
    if debug:
        print(f"  Valid sizes: {valid_sizes}")
        print(f"  Conv sizes (to avoid circular): {conv_sizes}")
    
    # Pad input and kernel to conv_sizes
    x_padded = x
    weight_padded = weight_flipped
    
    # Pad spatial dimensions to conv_sizes
    for i, (current_size, target_size) in enumerate(zip(spatial_dims, conv_sizes)):
        if current_size < target_size:
            pad_total = target_size - current_size
            pad = [0] * (2 * ndim)
            pad[2 * (ndim - 1 - i)] = 0  # left pad
            pad[2 * (ndim - 1 - i) + 1] = pad_total  # right pad
            x_padded = torch.nn.functional.pad(x_padded, pad)
    
    for i, (current_size, target_size) in enumerate(zip(kernel_dims, conv_sizes)):
        if current_size < target_size:
            pad_total = target_size - current_size
            pad = [0] * (2 * ndim)
            pad[2 * (ndim - 1 - i)] = 0  # left pad
            pad[2 * (ndim - 1 - i) + 1] = pad_total  # right pad
            weight_padded = torch.nn.functional.pad(weight_padded, pad)
    
    if debug:
        print(f"  X padded shape: {x_padded.shape}")
        print(f"  Weight padded shape: {weight_padded.shape}")
    
    # FFT both tensors
    x_fft = torch.fft.fftn(x_padded, dim=tuple(range(-ndim, 0)))
    w_fft = torch.fft.fftn(weight_padded, dim=tuple(range(-ndim, 0)))
    
    # Element-wise multiplication in frequency domain
    # Sum over input channels
    conv_fft = torch.einsum('bi...,oi...->bo...', x_fft, w_fft)
    conv_result = torch.fft.ifftn(conv_fft, dim=tuple(range(-ndim, 0))).real
    
    if debug:
        print(f"  Conv result shape: {conv_result.shape}")
    
    # Extract valid convolution region
    # Valid region starts at (kernel_size - 1) and has size equal to valid_sizes
    slices = [slice(None), slice(None)]
    for k, valid_size in zip(kernel_dims, valid_sizes):
        start_idx = k - 1
        slices.append(slice(start_idx, start_idx + valid_size))
    
    if debug:
        print(f"  Extraction slices: {slices}")
    
    output = conv_result[tuple(slices)]
    
    if debug:
        print(f"  Final output shape: {output.shape}")
    
    if bias is not None:
        output += bias.view(1, -1, *([1] * ndim))
    
    return output

class NDConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = kernel_size
        self.weight = None
        self.bias = nn.Parameter(torch.randn(out_ch))
    
    def forward(self, x):
        ndim = x.ndim - 2
        if self.weight is None:
            weight_shape = (self.out_ch, self.in_ch, *([self.k] * ndim))
            self.weight = nn.Parameter(torch.randn(*weight_shape, device=x.device))
        
        return fft_conv_nd(x, self.weight, self.bias, debug=False)

class NDConvNet(nn.Module):
    def __init__(self, input_shape, d_o, channels=[16, 32]):
        super().__init__()
        
        layers = []
        in_ch = input_shape[0]
        for out_ch in channels:
            layers.extend([NDConv(in_ch, out_ch), nn.ReLU()])
            in_ch = out_ch
        
        self.convs = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], d_o)
    
    def forward(self, x):
        x = self.convs(x)
        # Global average pooling
        x = x.mean(dim=tuple(range(2, x.ndim)))
        return self.fc(x)

# Numerical comparison functions
def compare_conv_implementations():
    """Compare FFT conv with standard PyTorch conv numerically"""
    print("=" * 60)
    print("NUMERICAL COMPARISON: FFT Conv vs PyTorch Conv")
    print("=" * 60)
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Test parameters
    batch_size, in_ch, out_ch = 2, 4, 8
    spatial_size = 16
    kernel_size = 3
    
    # Create test input
    x = torch.randn(batch_size, in_ch, spatial_size, spatial_size)
    
    # Create weights and bias (shared between implementations)
    weight = torch.randn(out_ch, in_ch, kernel_size, kernel_size)
    bias = torch.randn(out_ch)
    
    print(f"Test setup:")
    print(f"  Input shape: {x.shape}")
    print(f"  Weight shape: {weight.shape}")
    print(f"  Kernel size: {kernel_size}x{kernel_size}")
    print()
    
    # 1. Standard PyTorch Conv2d (no padding for valid convolution)
    torch_conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, bias=True)
    torch_conv.weight.data = weight.clone()
    torch_conv.bias.data = bias.clone()
    
    with torch.no_grad():
        torch_output = torch_conv(x)
    
    print(f"  PyTorch Conv2d output shape: {torch_output.shape}")
    print(f"  Expected output size: {spatial_size - kernel_size + 1}x{spatial_size - kernel_size + 1}")
    
    # 2. Our FFT implementation (with debug)
    print("  Running FFT convolution with debug info:")
    fft_output = fft_conv_nd(x, weight, bias, debug=True)
    print(f"  FFT output shape: {fft_output.shape}")
    print()
    
    # 3. Manual convolution for verification (slow but accurate)
    def manual_conv2d(x, weight, bias):
        B, C_in, H, W = x.shape
        C_out, _, K, K = weight.shape
        H_out, W_out = H - K + 1, W - K + 1
        
        output = torch.zeros(B, C_out, H_out, W_out)
        for b in range(B):
            for c_out in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        conv_sum = 0
                        for c_in in range(C_in):
                            for kh in range(K):
                                for kw in range(K):
                                    conv_sum += x[b, c_in, h+kh, w+kw] * weight[c_out, c_in, kh, kw]
                        output[b, c_out, h, w] = conv_sum + bias[c_out]
        return output
    
    manual_output = manual_conv2d(x, weight, bias)
    
    # Compare results
    print("Results comparison:")
    print(f"  PyTorch output shape: {torch_output.shape}")
    print(f"  FFT output shape: {fft_output.shape}")
    print(f"  Manual output shape: {manual_output.shape}")
    print()
    
    # Numerical differences
    diff_torch_fft = torch.abs(torch_output - fft_output)
    diff_torch_manual = torch.abs(torch_output - manual_output)
    diff_fft_manual = torch.abs(fft_output - manual_output)
    
    print("Absolute differences:")
    print(f"  PyTorch vs FFT:")
    print(f"    Mean: {diff_torch_fft.mean().item():.2e}")
    print(f"    Max:  {diff_torch_fft.max().item():.2e}")
    print(f"  PyTorch vs Manual:")
    print(f"    Mean: {diff_torch_manual.mean().item():.2e}")
    print(f"    Max:  {diff_torch_manual.max().item():.2e}")
    print(f"  FFT vs Manual:")
    print(f"    Mean: {diff_fft_manual.mean().item():.2e}")
    print(f"    Max:  {diff_fft_manual.max().item():.2e}")
    print()
    
    # Sample values for visual inspection
    print("Sample output values (first 3x3 of first channel, first batch):")
    print("PyTorch:")
    print(torch_output[0, 0, :3, :3])
    print("FFT:")
    print(fft_output[0, 0, :3, :3])
    print("Manual:")
    print(manual_output[0, 0, :3, :3])
    print()
    
    # Check if results are close
    torch_fft_close = torch.allclose(torch_output, fft_output, rtol=1e-4, atol=1e-6)
    torch_manual_close = torch.allclose(torch_output, manual_output, rtol=1e-4, atol=1e-6)
    fft_manual_close = torch.allclose(fft_output, manual_output, rtol=1e-4, atol=1e-6)
    
    print("Numerical equivalence (rtol=1e-4, atol=1e-6):")
    print(f"  PyTorch ≈ FFT: {torch_fft_close}")
    print(f"  PyTorch ≈ Manual: {torch_manual_close}")
    print(f"  FFT ≈ Manual: {fft_manual_close}")
    
    return torch_fft_close and fft_manual_close

def performance_comparison():
    """Compare performance of different convolution methods"""
    import time
    
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Test different sizes
    sizes = [(32, 32), (64, 64), (128, 128)]
    kernel_sizes = [3, 5, 7, 11]
    
    for spatial_size in sizes:
        print(f"\nSpatial size: {spatial_size[0]}x{spatial_size[1]}")
        print("-" * 40)
        
        for k in kernel_sizes:
            # Setup
            x = torch.randn(4, 8, *spatial_size)
            weight = torch.randn(16, 8, k, k)
            bias = torch.randn(16)
            
            # PyTorch conv
            torch_conv = nn.Conv2d(8, 16, k, padding=0, bias=True)
            torch_conv.weight.data = weight.clone()
            torch_conv.bias.data = bias.clone()
            
            # Warmup
            for _ in range(5):
                _ = torch_conv(x)
                _ = fft_conv_nd(x, weight, bias)
            
            # Time PyTorch
            torch_times = []
            for _ in range(20):
                start = time.time()
                with torch.no_grad():
                    _ = torch_conv(x)
                torch_times.append(time.time() - start)
            
            # Time FFT
            fft_times = []
            for _ in range(20):
                start = time.time()
                _ = fft_conv_nd(x, weight, bias)
                fft_times.append(time.time() - start)
            
            torch_avg = sum(torch_times) / len(torch_times) * 1000  # ms
            fft_avg = sum(fft_times) / len(fft_times) * 1000  # ms
            speedup = torch_avg / fft_avg if fft_avg > 0 else float('inf')
            
            print(f"  Kernel {k}x{k}: PyTorch={torch_avg:.2f}ms, FFT={fft_avg:.2f}ms, "
                  f"Speedup={speedup:.2f}x")

def nd_performance_comparison():
    """Compare performance for N-dimensional convolutions"""
    import time
    
    print("\n" + "=" * 60)
    print("N-DIMENSIONAL PERFORMANCE COMPARISON")
    print("=" * 60)
    
    test_configs = [
        # (ndim, spatial_size, kernel_size, description)
        (1, 256, 7, "1D: Audio/sequence processing"),
        (1, 1024, 15, "1D: Long sequences, large kernels"),
        (2, 64, 7, "2D: Standard image conv"),
        (2, 128, 15, "2D: Large kernel image processing"),
        (3, 32, 5, "3D: Video/medical imaging"), 
        (3, 64, 7, "3D: Large 3D volumes"),
        (4, 16, 3, "4D: Spatiotemporal"),
        (4, 24, 5, "4D: Larger spatiotemporal"),
        (5, 12, 3, "5D: High-dimensional data"),
    ]
    
    for ndim, spatial_size, kernel_size, description in test_configs:
        print(f"\n{description}")
        print(f"  {ndim}D convolution: {spatial_size}^{ndim} input, {kernel_size}^{ndim} kernel")
        print("-" * 50)
        
        # Create test tensors
        input_shape = [2, 4] + [spatial_size] * ndim  # batch=2, channels=4
        kernel_shape = [8, 4] + [kernel_size] * ndim  # out_ch=8, in_ch=4
        
        x = torch.randn(*input_shape)
        weight = torch.randn(*kernel_shape)
        bias = torch.randn(8)
        
        # Create equivalent PyTorch conv if available
        pytorch_conv = None
        if ndim == 1:
            pytorch_conv = torch.nn.Conv1d(4, 8, kernel_size, padding=0, bias=True)
        elif ndim == 2:
            pytorch_conv = torch.nn.Conv2d(4, 8, kernel_size, padding=0, bias=True)
        elif ndim == 3:
            pytorch_conv = torch.nn.Conv3d(4, 8, kernel_size, padding=0, bias=True)
        
        if pytorch_conv is not None:
            pytorch_conv.weight.data = weight.clone()
            pytorch_conv.bias.data = bias.clone()
        
        # Warmup
        for _ in range(3):
            if pytorch_conv is not None:
                with torch.no_grad():
                    _ = pytorch_conv(x)
            _ = fft_conv_nd(x, weight, bias, debug=False)
        
        # Time PyTorch (if available)
        pytorch_times = []
        if pytorch_conv is not None:
            for _ in range(10):
                start = time.time()
                with torch.no_grad():
                    pytorch_output = pytorch_conv(x)
                pytorch_times.append(time.time() - start)
            pytorch_avg = sum(pytorch_times) / len(pytorch_times) * 1000  # ms
        else:
            pytorch_avg = None
            pytorch_output = None
        
        # Time FFT
        fft_times = []
        for _ in range(10):
            start = time.time()
            fft_output = fft_conv_nd(x, weight, bias, debug=False)
            fft_times.append(time.time() - start)
        fft_avg = sum(fft_times) / len(fft_times) * 1000  # ms
        
        # Calculate theoretical complexity
        input_size = spatial_size ** ndim
        kernel_size_total = kernel_size ** ndim
        
        # Theoretical operations
        direct_ops = input_size * kernel_size_total  # Approximate
        fft_ops = input_size * ndim * torch.log2(torch.tensor(float(spatial_size)))  # Approximate
        
        theoretical_speedup = direct_ops / fft_ops if fft_ops > 0 else 1
        
        # Print results
        if pytorch_avg is not None:
            actual_speedup = pytorch_avg / fft_avg if fft_avg > 0 else float('inf')
            print(f"    PyTorch: {pytorch_avg:.2f}ms")
            print(f"    FFT:     {fft_avg:.2f}ms")
            print(f"    Speedup: {actual_speedup:.2f}x")
            
            # Verify numerical accuracy for available PyTorch convs
            if pytorch_output is not None and fft_output is not None:
                max_diff = torch.abs(pytorch_output - fft_output).max().item()
                print(f"    Max diff: {max_diff:.2e}")
        else:
            print(f"    FFT:     {fft_avg:.2f}ms")
            print(f"    (No PyTorch equivalent for {ndim}D)")
        
        print(f"    Theoretical speedup potential: {theoretical_speedup:.1f}x")
        print(f"    Total elements: {input_size:,}")

def large_kernel_comparison():
    """Test performance with very large kernels where FFT should excel"""
    import time
    
    print("\n" + "=" * 60) 
    print("LARGE KERNEL PERFORMANCE COMPARISON")
    print("=" * 60)
    
    configs = [
        (2, 128, [3, 7, 15, 31, 63], "2D: Image processing with large kernels"),
        (3, 64, [3, 7, 15, 31], "3D: Volume processing with large kernels"),
    ]
    
    for ndim, spatial_size, kernel_sizes, description in configs:
        print(f"\n{description}")
        print(f"  Input size: {spatial_size}^{ndim}")
        print("-" * 50)
        
        for kernel_size in kernel_sizes:
            # Create test tensors
            input_shape = [1, 4] + [spatial_size] * ndim
            kernel_shape = [8, 4] + [kernel_size] * ndim
            
            x = torch.randn(*input_shape)
            weight = torch.randn(*kernel_shape)
            bias = torch.randn(8)
            
            # PyTorch conv
            pytorch_conv = None
            if ndim == 2:
                pytorch_conv = torch.nn.Conv2d(4, 8, kernel_size, padding=0, bias=True)
            elif ndim == 3:
                pytorch_conv = torch.nn.Conv3d(4, 8, kernel_size, padding=0, bias=True)
            
            if pytorch_conv is not None:
                pytorch_conv.weight.data = weight.clone()
                pytorch_conv.bias.data = bias.clone()
            
            # Warmup
            for _ in range(2):
                if pytorch_conv is not None:
                    with torch.no_grad():
                        _ = pytorch_conv(x)
                _ = fft_conv_nd(x, weight, bias, debug=False)
            
            # Time both methods
            if pytorch_conv is not None:
                start = time.time()
                with torch.no_grad():
                    _ = pytorch_conv(x)
                pytorch_time = (time.time() - start) * 1000
            else:
                pytorch_time = None
            
            start = time.time()
            _ = fft_conv_nd(x, weight, bias, debug=False)
            fft_time = (time.time() - start) * 1000
            
            if pytorch_time is not None:
                speedup = pytorch_time / fft_time
                print(f"    Kernel {kernel_size}^{ndim}: PyTorch={pytorch_time:.1f}ms, "
                      f"FFT={fft_time:.1f}ms, Speedup={speedup:.2f}x")
            else:
                print(f"    Kernel {kernel_size}^{ndim}: FFT={fft_time:.1f}ms")

# Example usage and test
if __name__ == "__main__":
    # Test 1D convolution
    print("Testing 1D convolution:")
    x_1d = torch.randn(2, 3, 10)  # (batch, channels, length)
    model_1d = NDConvNet((3,), 5, channels=[8, 16])
    out_1d = model_1d(x_1d)
    print(f"Input shape: {x_1d.shape}, Output shape: {out_1d.shape}")
    
    # Test 2D convolution  
    print("\nTesting 2D convolution:")
    x_2d = torch.randn(2, 3, 32, 32)  # (batch, channels, height, width)
    model_2d = NDConvNet((3,), 5, channels=[8, 16])
    out_2d = model_2d(x_2d)
    print(f"Input shape: {x_2d.shape}, Output shape: {out_2d.shape}")
    
    # Test 3D convolution
    print("\nTesting 3D convolution:")
    x_3d = torch.randn(2, 3, 16, 16, 16)  # (batch, channels, depth, height, width)
    model_3d = NDConvNet((3,), 5, channels=[8, 16])
    out_3d = model_3d(x_3d)
    print(f"Input shape: {x_3d.shape}, Output shape: {out_3d.shape}")
    
    # Run numerical comparison
    comparison_passed = compare_conv_implementations()
    
    # Run 2D performance comparison
    performance_comparison()
    
    # Run N-D performance comparison
    nd_performance_comparison()
    
    # Run large kernel comparison
    large_kernel_comparison()
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: Numerical accuracy test {'PASSED' if comparison_passed else 'FAILED'}")
    print(f"{'='*60}")
