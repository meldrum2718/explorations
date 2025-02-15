import argparse

import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation


from .diffeq import DynSys, laplacian, gauss_ker2d
from ..utils import get_video_capture, get_appropriate_dims_for_ax_grid, inspect


def activation(x):
    return x
    # return x.clip(0, 1)
    # return F.tanh(x)


def main(args):
    for k, v in vars(args).items():
        print(k, v)

    H, W, C, K = args.H, args.W, args.C, args.K

    with torch.no_grad():

        cap = None
        try:

            cap = get_video_capture()

            ds = DynSys(H=H, W=W, C=C, K=K)
            inp = DynSys(H=H, W=W, C=C, K=K)

            ksize = 10
            xmin = -torch.pi
            xmax = torch.pi
            ymin = -torch.pi
            ymax = torch.pi
            mu1 = 0
            mu2 = 0
            s1 = 1; s2 = 0
            s3 = 1; s4 = 1
            mu = torch.Tensor([mu1, mu2])
            sigma = torch.Tensor([
                [s1, s2],
                [s3, s4]
            ])

            ker = gauss_ker2d(ksize, ksize, xmin, xmax, ymin, ymax, mu, sigma)
            log = laplacian(ker.reshape(1, 1, ksize, ksize))

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
                nonlocal s1
                nonlocal s2
                nonlocal s3
                nonlocal s4
                nonlocal ker
                nonlocal log

                t = 0

                mesh = torch.stack((
                    torch.linspace(0, 1, H).unsqueeze(-1).expand(H, W),
                    torch.linspace(0, 1, W).unsqueeze(0).expand(H, W)
                ), dim=0)
                pre_circle_mask = ((mesh - 0.5) ** 2).sum(dim=0) ** 0.5
 
                while True:
                    circle_mask = pre_circle_mask > bdy_radius
                    mu = torch.Tensor([mu1, mu2])
                    sigma = torch.Tensor([
                        [s1, s2],
                        [s3, s4]
                    ])
                    ker = gauss_ker2d(ksize, ksize, xmin, xmax, ymin, ymax, mu, sigma)
                    log = laplacian(ker.reshape(1, 1, ksize, ksize))
                    t += 1
                    _, frame = cap.read()
                    # convert frame to rgb with shape (h, w) and pixel values in [0, 1]
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (W, H), interpolation = cv2.INTER_AREA) / 255.0
                    frame = torch.Tensor(frame) # (H, W, C)
                    if C == 1:
                        frame = frame.mean(dim=-1).unsqueeze(-1) # make black and white
                    inp.step(frame, L=0)

                    for _ in range(its_per_step := 5):
                        L = 0
                        cur_state = ds.state[L][-1]
                        # ddt = wave_speed * laplacian(cur_state.unsqueeze(0).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).squeeze(0)
                        ddt = wave_speed * F.conv2d(
                            cur_state.unsqueeze(0).permute(0, 3, 1, 2),
                            log,
                            padding='same'
                        ).permute(0, 2, 3, 1).squeeze(0)

                        ds.step(ddt, L=L+2)

                        bdy_val = torch.Tensor([t * freq * (2 * torch.pi)]).sin().item()
                        for l in range(ds.K):
                            ds.state[l][circle_mask.unsqueeze(0).unsqueeze(-1).expand(ds.state[l].shape)] = bdy_val

                    yield t, ds

            ## set up figure for displaying the dyn.sys and some sliders
            fig, axs = plt.subplots(3, 3)
            axs = axs.reshape(-1)
            for ax in axs: ax.axis('off')
            fig.subplots_adjust(bottom=0.25)
            cmap = 'grey' if C == 1 else None
            inp_ims = [axs[k].imshow(activation(inp.state[k][-1]), cmap=cmap) for k in range(3)]
            inp_cbars = [plt.colorbar(im) for im in inp_ims]
            ds_ims = [axs[k + 3].imshow(activation(ds.state[k][-1]), cmap=cmap) for k in range(3)]
            ds_cbars = [plt.colorbar(im) for im in ds_ims]
            ker_im = axs[6].imshow(ker)
            ker_cbar = plt.colorbar(ker_im)

            log_im = axs[7].imshow(log.reshape(ksize, ksize))
            log_cbar = plt.colorbar(log_im)


            ## set up animation
            def draw_func(frame):
                nonlocal ker
                t, can = frame
                for k, (im, cbar) in enumerate(zip(inp_ims, inp_cbars)):
                    x = activation(inp.state[k][-1])
                    im.set_data(x)
                    im.set_clim(x.min(), x.max())
                    cbar.update_normal(im)
                    # im.set_clim(-2, 2)

                for k, (im, cbar) in enumerate(zip(ds_ims, ds_cbars)):
                    x = activation(ds.state[k][-1])
                    im.set_data(x)
                    im.set_clim(x.min(), x.max())
                    cbar.update_normal(im)
                    # im.set_clim(-2, 2)

                ker_im.set_data(ker)
                ker_im.set_clim(ker.min(), ker.max())
                ker_cbar.update_normal(ker_im)

                log_im.set_data(log.reshape(ksize, ksize))
                log_im.set_clim(log.min(), log.max())
                log_cbar.update_normal(log_im)



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

    args = parser.parse_args()

    assert args.C in (1, 3), 'require C == 1 or C == 3'

    main(args)
