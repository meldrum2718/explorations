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



## TODO wrt all the hyperparams particularly reln between step size and noise
## scales, i feel like there is a 'right' way of doing this, requires doing
## some math about convergence of dyn.sys i expect..



## TODO lets try to optimize.
## outer optimization over the graphs
## ## inner optimization over a batch dimenstion, stochastic perturbations of
#     the state ya know.. genetic algorithm like..

def main(args):
    cap = None
    try:

        if args.video_input:
            cap = get_video_capture()


        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        #
        #  leaving this approach for now for a different direction, more
        #  continuous! reocurring design principle in this project: if making
        #  unprincipled design decions, look for a way of making continuous and
        #  allowing it to be parameterized (sampled) by the tensor state.
        #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        ## ## # G = nx.barabasi_albert_graph(args.n_nodes, 3) ## TODO design decision
        ## ## # G = nx.turan_graph(args.n_nodes, 3) ## TODO design decision
        ## ## # G = nx.path_graph(args.n_nodes) ## TODO design decision
        ## ## # G = nx.random_geometric_graph(args.n_nodes, 0.2) ## TODO design decision
        ## ## G = nx.DiGraph(G)
        ## ## plot(G)


        can = CAN(
            B = args.batch_dim,
            C = args.channels,
            H = args.height,
            W = args.width,
            N = args.n_nodes,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
        )

        axh, axw = get_appropriate_dims_for_ax_grid(can.N)
        fig, axs = plt.subplots(axh, axw)
        axs = axs.reshape(-1)
        for ax in axs: ax.axis('off')

        cmap = None if (args.channels >= 3)else 'grey'

        ims = [axs[i].imshow(can.output(i)[0], cmap=cmap) for i in range(can.N)] # indexing into just first batch dim [0]

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
                    can.input(inp, node_idx=None)
        
                noise_scale = 1
                if args.use_noise_fbk:
                    if args.clean_period > 0:
                        noise_scale = (noise_fbk * torch.sin(torch.Tensor([t / args.clean_period]))**2).item()
                    can.add_noise(noise_scale)

                can.step(alpha=alpha * (1 - noise_scale)) ## TODO this is some arbirary (not a correct) way of varying step size .. todo be more principled about this

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

        ## TODO rethink this part of the impl., maybe rename noise_fbk to reflect new direction.
        ## initially i was planning on noise_fbk being a feedback signal that
        ## adds noise for misprediction. now think the ga select will cover this,
        ## so noise_fbk more properly named peak_noise_scale or something.
        ## but of course not crystalized on how adding noise to system. this
        ## sinusoidal noising process is something i feel like i want, i think its
        ## really a different noise scale than the noise that will drive the ga_selection
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
    parser.add_argument('--n_nodes', '-N', required=True, type=int) ## TODO later, based on thoughts i wrote on paper, would like to not specify N like this, but rather go back specifying n^k and working recursively across scales. but for now I am going to work with this impl.
    parser.add_argument('--wei_idx', required=False, default=None, type=int)
    parser.add_argument('--ker_idx', required=False, default=None, type=int)
    parser.add_argument('--stdin_idx', required=False, default=None, type=int)
    parser.add_argument('--stdout_idx', required=False, default=None, type=int)
    parser.add_argument('--clip_min', '-cmi', default=-2, type=float)
    parser.add_argument('--clip_max', '-cma', default=2, type=float)

    parser.add_argument('--video_input', action='store_true')
    parser.add_argument('--sample_period', '-sp', default=None, type=int)
    parser.add_argument('--alphamin', required=False, default='0', type=float)
    parser.add_argument('--alphamax', required=False, default='1', type=float)
    parser.add_argument('--noise_fbk_min', required=False, default=None, type=float)
    parser.add_argument('--noise_fbk_max', required=False, default=None, type=float)
    parser.add_argument('--clean_period', required=False, default=0, type=float)

    ## really might be interesting to sometimes work with discrete graphs, sometimes with these continuous ones.
    ## i suppose one way of doing this is overwriting can.state[can.wei_idx]
    ## at each step. really could be interesting to describe conv nets and the
    ## like in this language. should be very feasible. just need wei to have no
    ## edges going into it, so it doesnt change, and also have some conv kernels
    ## that also are fixed (having already been trained). ok, now thinking,
    ## there was something really nice about the rtn implementation with ceinsum
    ## operation that it was very flexible. ceinsum can represent a mat mul or
    ## a strided convolution. we like this.. i suppose we can sort of recover this
    ## by thinking of a sequence of mat muls (i.e. some matmuls for permuting,
    ## and then obviosly convolution is a linear operation that can be implemented
    ## as a matmul, just less efficiently than an einsum or a nn.conv2d, but
    ## still, the ease of adding a batch dim in this new impl seems sufficiently
    ## powerful so as to warrant this tradeoff)


    args = parser.parse_args()

    # validate args

    ## TODO really shouldnt have args.sample_period and args.clean_period. the 'correct' way of doing this (i expect) is just having one. for now ignoring this though
    ## TODO deal with this when implementing ga batch sel
    if args.sample_period:
        args.video_input = True
    if args.video_input:
        args.sample_period = args.sample_period or 1

    # if args.color:
    #     assert args.n == 3, 'handling color for n != 3 not currently supported.'

    args.use_noise_fbk = (None not in [args.noise_fbk_min, args.noise_fbk_max])



    main(args)
