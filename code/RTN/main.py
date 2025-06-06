import argparse

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

from .rtn import RTN
from ..utils import get_video_capture, rgb2grey, normalize, get_appropriate_dims_for_ax_grid, inspect


def main(args):
    cap = None
    try:
        n, k = args.n, args.k

        if args.video_input:
            cap = get_video_capture()

        h = w = n**k

        rtn = RTN(n=n,
                  k=k,
                  n_nodes=args.n_nodes,
                  batch_dim=args.batch_dim,
                  color=args.color,
                  clip_min=args.clip_min,
                  clip_max=args.clip_max,
        )

        axh, axw = get_appropriate_dims_for_ax_grid(args.n_nodes)
        fig, axs = plt.subplots(axh, axw)
        axs = axs.reshape(-1)
        for ax in axs: ax.axis('off')

        cmap = None if args.color else 'grey'

        state = normalize(torch.mean(rtn.nodes, dim=1)).detach().cpu().numpy() # average over batch dim, then normalize
        ims = [axs[i].imshow(state[i], cmap=cmap) for i in range(args.n_nodes)]

        alpha = 0
        noise_fbk = 0

        def step():
            nonlocal alpha
            nonlocal noise_fbk
            t = 0
            while True:
                t += 1
                inp = None
                if (args.sample_period is not None) and (t % args.sample_period == 0) and (cap is not None):
                    _, inp = cap.read()
                    # convert inp to rgb with shape (h, w) and pixel values in [0, 1]
                    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
                    inp = cv2.resize(inp, (w, h), interpolation = cv2.INTER_AREA) / 255.0
                    inp = torch.Tensor(inp) # (H, W, C)
                    if not args.color:
                        inp = inp
                        inp = torch.mean(inp, dim=0).unsqueeze(0)
                    # assert np.all(inp.shape == rtn.output().shape), f'inp shape: {inp.shape},   rtn.out.shape: {rtn.output().shape}'

                if inp is not None:
                    rtn.input(inp)

                rtn.step(alpha=alpha)

                if args.use_noise_fbk:
                    noise = noise_fbk * torch.randn_like(rtn.nodes)
                    rtn.nodes += noise

                yield t, rtn

        def draw_func(frame):
            ## for node in rtn.nodes: imshow(node.output()). 
            t, rtn = frame
            state = normalize(torch.mean(rtn.nodes, dim=1)).detach().cpu().numpy() # average over batch dim, then normalize
            for i in range(args.n_nodes):
                ims[i].set_data(state[i])
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
        alpha_slider = Slider(fig.add_axes([0.2, 0.05, 0.65, 0.03]), label=r'$\alpha$', valmin=args.alphamin, valmax=args.alphamax, valinit=alpha)
        alpha_slider.on_changed(update_alpha)

        if args.use_noise_fbk:
            def update_noise_fbk(x):
                nonlocal noise_fbk
                noise_fbk = x
            noise_fbk_slider = Slider(fig.add_axes([0.2, 0.10, 0.65, 0.03]), label='Noise feedback', valmin=args.noise_fbk_min, valmax=args.noise_fbk_max, valinit=noise_fbk)
            noise_fbk_slider.on_changed(update_noise_fbk)

        plt.show()

    finally:
        if cap is not None:
            cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            plt.cla()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', required=True, type=int)
    parser.add_argument('--k', required=True, type=int)
    parser.add_argument('--n_nodes', required=True, type=int)
    parser.add_argument('--video_input', action='store_true')
    parser.add_argument('--color', '-c', action='store_true')
    parser.add_argument('--clip_min', '-cmi', default=-2, type=float)
    parser.add_argument('--clip_max', '-cma', default=2, type=float)
    parser.add_argument('--sample_period', '-sp', default=None, type=int)
    parser.add_argument('--alphamin', required=False, default='0', type=float)
    parser.add_argument('--alphamax', required=False, default='1', type=float)
    parser.add_argument('--noise_fbk_min', required=False, default=None, type=float)
    parser.add_argument('--noise_fbk_max', required=False, default=None, type=float)
    parser.add_argument('--batch_dim', '-bd', default=1, type=int)


    args = parser.parse_args()

    # validate args
    if args.sample_period:
        args.video_input = True
    if args.video_input:
        args.sample_period = args.sample_period or 1

    if args.color:
        assert args.n == 3, 'handling color for n != 3 not currently supported.'

    args.use_noise_fbk = (None not in [args.noise_fbk_min, args.noise_fbk_max])



    main(args)
