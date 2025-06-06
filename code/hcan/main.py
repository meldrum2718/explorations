import argparse

import cv2
import torch
import numpy as np
from sympy import divisors
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

from .hcan import HCAN
from ..utils import get_video_capture, rgb2grey, normalize, get_appropriate_dims_for_ax_grid, inspect, plot


def main(args):
    for k, v in vars(args).items():
        print(k, v)

    with torch.no_grad():

        cap = None
        try:

            if args.video_input:
                cap = get_video_capture()

            hcan = HCAN(
                N=args.N,
                K=args.K,
                C=args.C,
            )

            alpha = 0.1
            noise_scale = 0
            L_step = 0
            L_input = 1
            input_alpha = 1

            H = W = args.N ** args.K


            def step():
                nonlocal alpha
                nonlocal L_step
                nonlocal L_input
                nonlocal noise_scale
                nonlocal input_alpha

                t = 0

                while True:
                    t += 1
                    inp = torch.zeros(H, W, 1)
                    if args.video_input:
                        _, inp = cap.read()
                        # convert inp to rgb with shape (h, w) and pixel values in [0, 1]
                        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
                        inp = cv2.resize(inp, (W, H), interpolation = cv2.INTER_AREA) / 255.0
                        inp = torch.Tensor(inp) # (H, W, C)
                        inp = inp.mean(dim=-1).unsqueeze(-1) # make black and white


                        L_inp = L_input
                        if args.use_random_L_input:
                            L_inp = torch.randint(0, L_input+1, size=(1,)).item()

                        input_idx = args.stdin_idx
                        if args.use_random_input_idx:
                            idx_max = args.N**(2*L_inp) if L_inp > 0 else 1
                            input_idx = torch.randint(0, idx_max, size=(1,)).item()

                        hcan.input(
                            inp,
                            node_idx=input_idx,
                            L=L_inp,
                            alpha=input_alpha,
                        )

                    if args.use_noise_fbk:
                        hcan.add_noise(alpha * noise_scale) # * pred_error

                    L_st = L_step
                    if args.use_random_L_step and L_step > 0:
                        L_step_sampler = torch.distributions.categorical.Categorical(logits=L_step - torch.arange(L_step) + 1)
                        L_st = L_step_sampler.sample()
                    print('L_st:', L_st)
                    hcan.step(alpha=alpha, L=t % (L_step+1), C_min=0, C_max=None) ## TODO guesswork hacky L_step schedule
                    
                    yield t, hcan


            ## set up figure for displaying the dyn.sys and some sliders
            fig, axs = plt.subplots(1, hcan.C)
            if not hasattr(axs, '__iter__'):
                axs = [axs]
            for ax in axs: ax.axis('off')
            fig.subplots_adjust(bottom=0.25)
            cmap = 'grey'
            ims = [axs[ci].imshow(hcan.state[..., ci], cmap=cmap) for ci in range(hcan.C)]

            ## set up animation
            def draw_func(frame):
                t, can = frame
                for ci, im in enumerate(ims):
                    im.set_data(hcan.state[..., ci])

                fig.suptitle(str(t))
                return ims

            ani = FuncAnimation(
                fig,
                func=draw_func,
                frames=step,
                interval=10,
                save_count=1,
            )


            def update_alpha(x):
                nonlocal alpha
                alpha = x
            alpha_slider = Slider(fig.add_axes((0.2, 0.03, 0.65, 0.03)), label=r'$\alpha$', valmin=args.alphamin, valmax=args.alphamax, valinit=alpha)
            alpha_slider.on_changed(update_alpha)


            def update_L_input(x):
                nonlocal L_input
                L_input = x
            L_input_slider = Slider(fig.add_axes((0.2, 0.06, 0.65, 0.03)), label='L_input', valmin=0, valmax=args.K-1, valinit=0, valstep=1)
            L_input_slider.on_changed(update_L_input)


            def update_L_step(x):
                nonlocal L_step
                L_step = x
            L_step_slider = Slider(fig.add_axes((0.2, 0.09, 0.65, 0.03)), label='L_step', valmin=0, valmax=args.K-1, valinit=0, valstep=1)
            L_step_slider.on_changed(update_L_step)


            def update_input_alpha(x):
                nonlocal input_alpha
                input_alpha = x
            input_alpha_slider = Slider(fig.add_axes((0.2, 0.12, 0.65, 0.03)), label=r'input $\alpha$', valmin=0, valmax=1, valinit=input_alpha)
            input_alpha_slider.on_changed(update_input_alpha)


            if args.use_noise_fbk:

                def update_noise_scale(x):
                    nonlocal noise_scale
                    noise_scale = x
                noise_fbk_slider = Slider(fig.add_axes((0.2, 0.15, 0.65, 0.03)), label='Noise scale', valmin=args.noise_fbk_min, valmax=args.noise_fbk_max, valinit=noise_scale)
                noise_fbk_slider.on_changed(update_noise_scale)

            plt.show()

        finally:
            if cap is not None:
                cap.release()
                cv2.destroyAllWindows()
                cv2.waitKey(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-N', required=True, type=int)
    parser.add_argument('-K', required=True, type=int)
    parser.add_argument('-C', required=True, type=int)

    ## not using these for now.. instead just updating default args in CAN.__init__
    ## parser.add_argument('--wei_idx', required=False, default=None, type=int)
    ## parser.add_argument('--ker_idx', required=False, default=None, type=int)
    parser.add_argument('--stdin_idx', required=False, default=None, type=int)
    ## parser.add_argument('--stdout_idx', required=False, default=None, type=int)

    parser.add_argument('--use_random_input_idx', action='store_true')
    parser.add_argument('--use_random_L_input', action='store_true')
    parser.add_argument('--use_random_L_step', action='store_true')

    parser.add_argument('--video_input', action='store_true')

    parser.add_argument('--sample_period', '-sp', default=1, type=int)

    parser.add_argument('--alphamin', required=False, default='0', type=float)
    parser.add_argument('--alphamax', required=False, default='1', type=float)

    parser.add_argument('--noise_fbk_min', required=False, default=None, type=float)
    parser.add_argument('--noise_fbk_max', required=False, default=None, type=float)

    ## TODO
    ## parser.add_argument('--activation')
    ## parser.add_argument('--step_implementation')



    args = parser.parse_args()

    # validate args

    args.use_noise_fbk = (None not in [args.noise_fbk_min, args.noise_fbk_max])

    main(args)
