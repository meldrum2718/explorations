import argparse

import cv2
import torch
import colorsys
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation


from ..utils import get_video_capture, get_appropriate_dims_for_ax_grid, inspect


class ComplexDynamicalSystem:
    def __init__(
        self,
        H: int,
        W: int,
        C: int,
        K: int,
    ):

        self.H = H
        self.W = W
        self.C = C
        self.K = K

        self.state = [torch.zeros(K - L + 1, H, W, C, dtype=torch.complex64) for L in range(K + 1)] # pyramid like

    def step(self, im, L=0):
        assert im.shape == (self.H, self.W, self.C)

        x = self.state[L].roll(shifts=(-1,), dims=(0,))
        x[-1] = im
        self.state[L] = x

        if L > 0:
            # propegate updards
            for k in range(L - 1, -1, -1): # [L-1, ..., 0]
                self.state[k] = self.state[k].roll(shifts=(-1,), dims=(0,))
                self.state[k][-1] = self.state[k][-2] + self.state[k+1][-1]

        if L < self.K:
            # propegate downwards
            for k in range(L + 1, self.K + 1): # [L+1, ..., K]
                self.state[k] = self.state[k].roll(shifts=(-1,), dims=(0,))
                self.state[k][-1] = self.state[k-1][-1] - self.state[k-1][-2]

    def resize(self, h, w):
        resized = ComplexDynamicalSystem(h, w, self.C, self.K)
        for k in range(self.K + 1):
            # resized.state[k] = F.interpolate(self.state[k].permute(0, 3, 1, 2), size=(h, w), mode='bilinear', antialias='true').permute(0, 2, 3, 1) 
            resized.state[k] = bilinear_interpolate_complex(self.state[k], h, w)
        return resized

    def normalize(self):
        for k in range(len(self.state)):
            x = self.state[k]

            min_val = x.abs().min()  # Minimum magnitude
            x_shifted = x - min_val  # Shift by minimum magnitude
            max_val = x_shifted.abs().max()  # Maximum magnitude after shifting

            self.state[k] = x_shifted / max_val if max_val != 0 else x
    
            # self.state[k] /= (self.state[k].real.max() + 1e-9)



Dx = torch.tensor([-1, 1]).reshape(1, 1, -1).to(torch.complex64)
D2x = F.conv1d(Dx, Dx.flip(-1), padding=1).unsqueeze(0)
D2y = D2x.transpose(-1, -2)



def bilinear_interpolate_complex(z_tensor, new_h, new_w):
    """
    Performs bilinear interpolation on a batched complex-valued PyTorch tensor.
    
    Parameters:
    - z_tensor: Input PyTorch complex tensor of shape (B, H, W, C)
    - new_h: Target height
    - new_w: Target width
    
    Returns:
    - Interpolated complex tensor of shape (B, new_h, new_w, C)
    """
    B, H, W, C = z_tensor.shape
    
    # Create normalized coordinate grids
    y = torch.linspace(0, H - 1, new_h, device=z_tensor.device)
    x = torch.linspace(0, W - 1, new_w, device=z_tensor.device)
    
    # Get integer pixel indices
    y0 = torch.floor(y).long()
    x0 = torch.floor(x).long()
    y1 = torch.clamp(y0 + 1, 0, H - 1)
    x1 = torch.clamp(x0 + 1, 0, W - 1)
    
    # Get fractional parts
    wy1 = y - y0.float()
    wx1 = x - x0.float()
    wy0 = 1 - wy1
    wx0 = 1 - wx1
    
    # Expand dimensions for broadcasting
    (y0, x0), (y1, x1) = torch.meshgrid(y0, x0, indexing='ij'), torch.meshgrid(y1, x1, indexing='ij')
    
    # Expand batch and channel dimensions for broadcasting
    y0, x0, y1, x1 = [t.expand(B, -1, -1, C) for t in [y0, x0, y1, x1]]
    
    # Perform bilinear interpolation separately on real and imaginary parts
    real = (z_tensor.real[:, y0, x0, :] * (wy0 * wx0).unsqueeze(-1) +
            z_tensor.real[:, y0, x1, :] * (wy0 * wx1).unsqueeze(-1) +
            z_tensor.real[:, y1, x0, :] * (wy1 * wx0).unsqueeze(-1) +
            z_tensor.real[:, y1, x1, :] * (wy1 * wx1).unsqueeze(-1))
    
    imag = (z_tensor.imag[:, y0, x0, :] * (wy0 * wx0).unsqueeze(-1) +
            z_tensor.imag[:, y0, x1, :] * (wy0 * wx1).unsqueeze(-1) +
            z_tensor.imag[:, y1, x0, :] * (wy1 * wx0).unsqueeze(-1) +
            z_tensor.imag[:, y1, x1, :] * (wy1 * wx1).unsqueeze(-1))
    
    return torch.complex(real, imag)






def laplacian(im):

    assert im.ndim == 4, 'im should be a (B, C, H, W) tensor'

    return torch.sum(
        torch.stack((
            F.conv2d(im, D2x, padding='same'),
            F.conv2d(im, D2y, padding='same')),
            dim=0),
        dim=0)


def mesh2d(H, W, xmin=0, xmax=1, ymin=0, ymax=1, use_xy_format=False):
    i = torch.linspace(ymin, ymax, H).reshape(H, 1).expand(H, W)
    j = torch.linspace(xmin, xmax, W).reshape(1, W).expand(H, W)
    if use_xy_format:
        return torch.stack((j, i))
    return torch.stack((i, j), dim=-1)


def gauss_ker2d(nh, nw, xmin, xmax, ymin, ymax, mu, sigma):    
    mesh = mesh2d(nh, nw, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    sigma_inv = torch.inverse(sigma)
    
    flat_mesh = mesh.reshape(-1, 2)
    centered_mesh = flat_mesh - mu
    quad_form = torch.sum(centered_mesh @ sigma_inv * centered_mesh, dim=1)
    return  torch.exp(-0.5 *1j * quad_form).reshape(nh, nw
                                               ) / (
                                                   2 * torch.pi * torch.linalg.det(sigma) ** 0.5)


def activation(x):
    return x
    # return x.clip(0, 1)
    # return F.tanh(x)


def complex_to_rgb(z_array, max_magnitude=1.0):
    # return z_array.real
    """
    Maps a PyTorch tensor of complex numbers to an RGB color tensor.
    
    Parameters:
    - z_array: torch tensor of complex numbers
    - max_magnitude: maximum magnitude for normalization (adjust based on your data range)
    
    Returns:
    - torch tensor of shape (..., 3) with RGB values in range [0, 255]
    """
    magnitude = torch.abs(z_array)
    phase = torch.angle(z_array)  # Phase (angle) in radians
    
    # Normalize magnitude to [0, 1] for brightness control
    brightness = torch.clamp(magnitude / max_magnitude, 0.0, 1.0)
    
    # Convert phase from [-pi, pi] to [0, 1] for Hue mapping
    hue = (phase + torch.pi) / (2 * torch.pi)
    
    # Convert HSV to RGB
    rgb_list = [colorsys.hsv_to_rgb(h.item(), 1.0, b.item()) for h, b in zip(hue.flatten(), brightness.flatten())]
    
    # Convert to PyTorch tensor and reshape to original shape with an extra dimension for RGB
    rgb_tensor = torch.tensor(rgb_list, dtype=torch.float32).reshape(*z_array.shape, 3) * 255
    
    return rgb_tensor.int()


def main(args):
    for k, v in vars(args).items():
        print(k, v)

    H, W, C, K = args.H, args.W, args.C, args.K

    with torch.no_grad():

        cap = None
        try:

            # cap = get_video_capture()

            ds = ComplexDynamicalSystem(H=H, W=W, C=C, K=K)

            sigma_min = -1
            sigma_max = 1

            ksize = 1
            xmin = -torch.pi
            xmax = torch.pi
            ymin = -torch.pi
            ymax = torch.pi
            mu1 = 0
            mu2 = 0
            s11 = 1; s12 = 0
            mu = torch.Tensor([mu1, mu2])
            sigma = torch.Tensor([
                [s11, s12],
                [s12, s11]
            ])

            ker = gauss_ker2d(ksize, ksize, xmin, xmax, ymin, ymax, mu, sigma)
            lap = laplacian(ker.reshape(1, 1, ksize, ksize))

            # alpha_camera = 1
            freq = 1
            wave_speed = 0.4
            bdy_radius = 0.5

            def step():
                # nonlocal alpha_camera
                nonlocal freq
                nonlocal wave_speed
                nonlocal bdy_radius
                nonlocal ksize
                nonlocal s11
                nonlocal s12
                nonlocal mu1
                nonlocal mu2
                nonlocal ker
                nonlocal lap
                nonlocal ds
                nonlocal H
                nonlocal W

                t = 0
 
                while True:
                    if (H != ds.H) or (W != ds.W):
                        ds = ds.resize(H, W)

                    mesh = torch.stack((
                        torch.linspace(0, 1, H).unsqueeze(-1).expand(H, W),
                        torch.linspace(0, 1, W).unsqueeze(0).expand(H, W)
                    ), dim=0)
                    pre_circle_mask = ((mesh - 0.5) ** 2).sum(dim=0) ** 0.5
                    circle_mask = pre_circle_mask > bdy_radius
                    mu = torch.Tensor([mu1, mu2])
                    sigma = torch.Tensor([
                        [s11, s12],
                        [s12, s11]
                    ])
                    ker = gauss_ker2d(ksize, ksize, xmin, xmax, ymin, ymax, mu, sigma).to(torch.complex64)
                    lap = laplacian(ker.reshape(1, 1, ksize, ksize))
                    t += 1
                    # _, frame = cap.read()
                    # convert frame to rgb with shape (h, w) and pixel values in [0, 1]
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # frame = cv2.resize(frame, (W, H), interpolation = cv2.INTER_AREA) / 255.0
                    # frame = torch.Tensor(frame) # (H, W, C)
                    # if C == 1:
                    #     frame = frame.mean(dim=-1).unsqueeze(-1) # make black and white
                    # inp.step(frame, L=0)

                    for _ in range(its_per_step := 1):
                        L = 0
                        cur_state = ds.state[L][-1]
                        # ddt = wave_speed * laplacian(cur_state.unsqueeze(0).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).squeeze(0)
                        ddt = 0.11j * wave_speed * F.conv2d(
                            cur_state.unsqueeze(0).permute(0, 3, 1, 2),
                            lap,
                            padding='same'
                        ).permute(0, 2, 3, 1).squeeze(0)

                        ds.step(ddt, L=L+2)
                        ds.normalize()

                        bdy_val = torch.Tensor([t * freq * (2 * torch.pi)]).sin().item()
                        for l in range(ds.K):
                            ds.state[l][
                                circle_mask.unsqueeze(0).unsqueeze(-1).expand(ds.state[l].shape)
                            ] = bdy_val

                    yield t, ds

            ## set up figure for displaying the dyn.sys and some sliders
            fig, axs = plt.subplots(2, 3)
            axs = axs.reshape(-1)
            for ax in axs: ax.axis('off')
            fig.subplots_adjust(bottom=0.25)
            cmap = 'grey' if C == 1 else None
            # inp_ims = [axs[k].imshow(activation(inp.state[k][-1]), cmap=cmap) for k in range(3)]
            # inp_cbars = [plt.colorbar(im) for im in inp_ims]

            ds_ims = [axs[k].imshow(complex_to_rgb(activation(ds.state[k][-1].squeeze(-1))), cmap=cmap) for k in range(3)]
            ds_cbars = [plt.colorbar(im) for im in ds_ims]
            ker_im = axs[3].imshow(complex_to_rgb(ker))
            ker_cbar = plt.colorbar(ker_im)

            lap_im = axs[4].imshow(complex_to_rgb(lap.reshape(ksize, ksize)))
            lap_cbar = plt.colorbar(lap_im)


            ## set up animation
            def draw_func(frame):
                nonlocal ker
                t, can = frame
                # for k, (im, cbar) in enumerate(zip(inp_ims, inp_cbars)):
                #     x = activation(inp.state[k][-1])
                #     im.set_data(x)
                #     im.set_clim(x.min(), x.max())
                #     cbar.update_normal(im)
                #     # im.set_clim(-2, 2)

                for k, (im, cbar) in enumerate(zip(ds_ims, ds_cbars)):
                    x = activation(ds.state[k][-1]).squeeze(-1)
                    xrgb = complex_to_rgb(x)
                    # inspect('x', x.real)
                    # inspect('xrgb', xrgb)
                    im.set_data(xrgb)
                    im.set_clim(xrgb.min(), xrgb.max())
                    cbar.update_normal(im)
                    # im.set_clim(-2, 2)

                ker_im.set_data(complex_to_rgb(ker))
                ker_im.set_clim(ker.real.min(), ker.real.max())
                ker_cbar.update_normal(ker_im)

                lap_im.set_data(complex_to_rgb(lap.reshape(ksize, ksize)))
                # lap_im.set_clim(lap.min(), lap.max())
                lap_cbar.update_normal(lap_im)

                fig.suptitle(str(t))
                return ds_ims

            ani = FuncAnimation(
                fig,
                func=draw_func,
                frames=step,
                interval=10,
                save_count=1,
            )


            # def update_alpha_camera(x):
            #     nonlocal alpha_camera
            #     alpha_camera = x
            # alpha_camera_slider = Slider(fig.add_axes((0.2, 0.03, 0.65, 0.03)), label='alpha Camera Input', valmin=0, valmax=1, valinit=1)
            # alpha_camera_slider.on_changed(update_alpha_camera)

            def update_freq(x):
                nonlocal freq
                freq = x
            freq_slider = Slider(fig.add_axes((0.2, 0.06, 0.65, 0.03)), label='freq', valmin=0, valmax=1, valinit=freq)
            freq_slider.on_changed(update_freq)

            def update_wave_speed(x):
                nonlocal wave_speed
                wave_speed = x
            wave_speed_slider = Slider(fig.add_axes((0.2, 0.09, 0.65, 0.03)), label='wave speed', valmin=0, valmax=1, valinit=wave_speed)
            wave_speed_slider.on_changed(update_wave_speed)

            def update_bdy_radius(x):
                nonlocal bdy_radius
                bdy_radius = x
            bdy_radius_slider = Slider(fig.add_axes((0.2, 0.12, 0.65, 0.03)), label='bdy_radius', valmin=0, valmax=1, valinit=bdy_radius)
            bdy_radius_slider.on_changed(update_bdy_radius)

            def update_ksize(x):
                nonlocal ksize
                ksize = 2 * int(x) + 1
            ksize_slider = Slider(fig.add_axes((0.2, 0.15, 0.65, 0.03)), label='ksize', valmin=1, valmax=20, valstep=1, valinit=ksize)
            ksize_slider.on_changed(update_ksize)

            def update_s11(x):
                nonlocal s11
                s11 = x
            s11_slider = Slider(fig.add_axes((0.2, 0.18, 0.65, 0.03)), label='s11', valmin=sigma_min, valmax=sigma_max, valinit=s11)
            s11_slider.on_changed(update_s11)

            def update_s12(x):
                nonlocal s12
                s12 = x
            s12_slider = Slider(fig.add_axes((0.2, 0.21, 0.65, 0.03)), label='s12', valmin=sigma_min, valmax=sigma_max, valinit=s12)
            s12_slider.on_changed(update_s12)


            def update_H(x):
                nonlocal H
                H = x
            H_slider = Slider(fig.add_axes((0.2, 0.24, 0.65, 0.03)), label='H', valmin=10, valmax=args.hmax, valstep=1, valinit=H)
            H_slider.on_changed(update_H)


            def update_W(x):
                nonlocal W
                W = x
            W_slider = Slider(fig.add_axes((0.2, 0.27, 0.65, 0.03)), label='W', valmin=10, valmax=args.wmax, valstep=1, valinit=W)
            W_slider.on_changed(update_W)


            plt.show()

        finally:
            if cap is not None:
                cap.release()
                cv2.destroyAllWindows()
                cv2.waitKey(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-H', required=True, type=int)
    parser.add_argument('-W', required=True, type=int)
    parser.add_argument('-K', required=True, type=int)
    parser.add_argument('-C', required=True, type=int)

    parser.add_argument('-wmax', default=1000, required=False, type=int)
    parser.add_argument('-hmax', default=1000, required=False, type=int)

    args = parser.parse_args()

    assert args.C in (1, 3), 'require C == 1 or C == 3'

    main(args)
