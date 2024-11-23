import argparse

import cv2
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

from .coupled_attn_net import CAN
from ..utils import get_video_capture, rgb2grey, normalize, get_appropriate_dims_for_ax_grid, inspect, plot



## TODO lets try to optimize.
## outer optimization over the graphs
## ## inner optimization over a batch dimenstion, stochastic perturbations of
#     the state ya know.. genetic algorithm like..

def main(args):
    cap = None
    try:

        if args.video_input:
            cap = get_video_capture()

        G = nx.barabasi_albert_graph(args.n_nodes, 3) ## TODO design decision
        # G = nx.turan_graph(args.n_nodes, 3) ## TODO design decision
        # G = nx.path_graph(args.n_nodes) ## TODO design decision
        # G = nx.random_geometric_graph(args.n_nodes, 0.2) ## TODO design decision
        G = nx.DiGraph(G)
        plot(G)

        can = CAN(
            B = args.batch_dim,
            C = args.channels,
            H = args.height,
            W = args.width,
            G=G,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
        )

        axh, axw = get_appropriate_dims_for_ax_grid(can.n_nodes)
        fig, axs = plt.subplots(axh, axw)
        axs = axs.reshape(-1)
        for ax in axs: ax.axis('off')

        cmap = None if (args.channels >= 3)else 'grey'

        ims = [axs[i].imshow(can.output(i)[0], cmap=cmap) for i in range(can.n_nodes)] # indexing into just first batch dim [0]

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
                    inp = cv2.resize(inp, (args.width, args.height), interpolation = cv2.INTER_AREA) / 255.0
                    inp = torch.Tensor(inp) # (H, W, C)
                    # if not args.color:
                    #     inp = inp
                    #     inp = torch.mean(inp, dim=0).unsqueeze(0)
                    # assert np.all(inp.shape == can.output().shape), f'inp shape: {inp.shape},   can.out.shape: {can.output().shape}'

                if inp is not None:
                    can.input(inp, node_idx=0)

                can.step(alpha=alpha)

                if args.use_noise_fbk:
                    if args.clean_period > 0:
                        noise = noise_fbk * torch.sin(torch.Tensor([t / args.clean_period]))**2 * torch.randn_like(can.state)
                    else:
                        noise = noise_fbk * torch.randn_like(can.state)
                    can.state += noise

                yield t, can

        def draw_func(frame):
            t, can = frame
            for i in range(args.n_nodes):
                ims[i].set_data(can.output(i)[0]) # indexing into just first batch dim [0]
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
    parser.add_argument('--height', '-H', required=True, type=int)
    parser.add_argument('--width', '-W', required=True, type=int)
    parser.add_argument('--batch_dim', '-B', required=True, type=int)
    parser.add_argument('--channels', '-C', required=True, type=int)
    parser.add_argument('--n_nodes', required=True, type=int)
    parser.add_argument('--video_input', action='store_true')
    # parser.add_argument('--color', '-c', action='store_true')
    parser.add_argument('--clip_min', '-cmi', default=-2, type=float)
    parser.add_argument('--clip_max', '-cma', default=2, type=float)
    parser.add_argument('--sample_period', '-sp', default=None, type=int)
    parser.add_argument('--alphamin', required=False, default='0', type=float)
    parser.add_argument('--alphamax', required=False, default='1', type=float)
    parser.add_argument('--noise_fbk_min', required=False, default=None, type=float)
    parser.add_argument('--noise_fbk_max', required=False, default=None, type=float)
    parser.add_argument('--clean_period', required=False, default=0, type=float)

    ## TODO add cli args for specifying type of graph.. for now just random BA graphs


    args = parser.parse_args()

    # validate args
    if args.sample_period:
        args.video_input = True
    if args.video_input:
        args.sample_period = args.sample_period or 1

    # if args.color:
    #     assert args.n == 3, 'handling color for n != 3 not currently supported.'

    args.use_noise_fbk = (None not in [args.noise_fbk_min, args.noise_fbk_max])



    main(args)
