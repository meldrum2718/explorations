import argparse

import cv2
import torch
import numpy as np
from sympy import divisors
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

from .coupled_attn_net import CAN
from ..utils import get_video_capture, rgb2grey, normalize, get_appropriate_dims_for_ax_grid, inspect, plot

## TODO wrt all the hyperparams particularly reln between step size and noise
## scales, i feel like there is a 'right' way of doing this, requires doing
## some math about convergence of dyn.sys i expect..

## TODO lets try to optimize.
## outer optimization over the graphs
## ## inner optimization over a batch dimenstion, stochastic perturbations of
#     the state ya know.. genetic algorithm like..

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


def main(args):
    for k, v in vars(args).items():
        print(k, v)

    with torch.no_grad(): ## TODO placement of this ?

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
                use_complex=args.complex,
            )

            alpha = 0
            noise_scale = 0
            ga_sel_period = args.ga_sel_period_max # initialize to longest selection period
            ga_topk = args.ga_topk_max # initialize to least selective topk
            ga_noise_scale = args.ga_noise_scale_min # initalize to least perturbative ga noise scale
            ga_scores = torch.zeros(args.batch_dim)

            def step():
                nonlocal alpha
                nonlocal noise_scale
                nonlocal ga_sel_period
                nonlocal ga_topk
                nonlocal ga_noise_scale
                nonlocal ga_scores

                t = 0
                while True:
                    t += 1
                    inp = torch.zeros(args.height, args.width, 3 if (args.channels >= 3) else 1)
                    if args.video_input:
                        if (t % args.sample_period) == 0:
                            _, inp = cap.read()
                            # convert inp to rgb with shape (h, w) and pixel values in [0, 1]
                            inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
                            inp = cv2.resize(inp, (args.width, args.height), interpolation = cv2.INTER_AREA) / 255.0
                            inp = torch.Tensor(inp) # (H, W, C)
                            if args.channels < 3:
                                inp = inp.mean(dim=-1).unsqueeze(-1)

                        inp_noise_scale = (t % args.sample_period) / args.sample_period
                        noise = torch.randn_like(inp) * inp_noise_scale
                        noisy_inp = inp + noise

                        can.input(noisy_inp, node_idx=None, ga_eval=args.use_ga)

                    if args.use_ga and (t % ga_sel_period) == 0:
                        ga_scores = can.ga_step(k=ga_topk, max_noise_scale=ga_noise_scale)

                    step_size = alpha
                    ns = step_size * noise_scale

                    if args.use_noise_fbk:

                        ## TODO clean period should be coupled with sample_period ..
                        if args.clean_period > 0:
                            t_osscil = torch.sin(torch.Tensor([t / args.clean_period])) ** 2 # time param for interpolation between diffeq step and noising process
                            t_osscil = t_osscil.item()
                            ns *= t_osscil

                        can.add_noise(ns)

                    can.step(alpha=step_size)
                    
                    yield t, can


            ## set up figure for displaying the dyn.sys and some sliders
            axh, axw = get_appropriate_dims_for_ax_grid(can.N) ## TODO given state is one big tensor, could really just display one big tensor in a single ax. but this would require a change to can.output, given its working at the moment will leave as is for now
            fig, axs = plt.subplots(axh, axw)
            axs = axs.reshape(-1)
            for ax in axs: ax.axis('off')
            fig.subplots_adjust(bottom=0.25)

            # axs[can.ker_idx].set_title('Ker')
            if can.query_idx is not None:
                axs[can.query_idx].set_title('Queries')
            if can.key_idx is not None:
                axs[can.key_idx].set_title('Keys')
            if can.value_idx is not None:
                axs[can.value_idx].set_title('Values')

            axs[can.wei_idx].set_title('Wei')
            axs[can.stdin_idx].set_title('Inp')
            axs[can.stdout_idx].set_title('Out')

            cmap = None if (args.channels >= 3) else 'grey'

            ims = [axs[i].imshow(can.output(i)[0], cmap=cmap) for i in range(can.N)] # indexing into just first batch dim [0] ## could try taking mean across batch dim..


            ## set up animation
            def draw_func(frame):
                t, can = frame
                for i in range(args.n_nodes):
                    ims[i].set_data(can.output(i)[0]) # indexing into just the first batch
                fig.suptitle(str(t))
                return ims

            ani = FuncAnimation(
                fig,
                func=draw_func,
                frames=step,
                interval=10,
                save_count=1,
            )


            if args.use_ga: ## set up GA score observations

                ga_fig, ga_ax = plt.subplots()
                bar_width = 0.5
                bars = [Rectangle((idx - bar_width / 2, 0), bar_width, 0, color='blue') for idx in range(args.batch_dim)]
                for bar in bars:
                    ga_ax.add_patch(bar)
                ga_ax.set_xlim(-1, args.batch_dim)
                ga_ax.set_title("GA scores")
                ga_ax.set_xlabel("Batch")
                ga_ax.set_ylabel("Prediction Error")

                def generate_ga_scores():
                    nonlocal ga_scores
                    while True:
                        yield ga_scores

                def ga_score_draw_func(ga_scores):
                    ga_ax.set_ylim(0, 1.5 * ga_scores.max() + 1e-9)
                    for bar, score in zip(bars, ga_scores):
                        bar.set_height(score)
                    return bars

                ga_score_ani = FuncAnimation(
                    ga_fig,
                    func=ga_score_draw_func,
                    frames=generate_ga_scores,
                    interval=10,
                    save_count=1,
                )


            ## set up sliders for interactively changing some hyperparams

            def update_alpha(x):
                nonlocal alpha
                alpha = x
            alpha_slider = Slider(fig.add_axes((0.2, 0.03, 0.65, 0.03)), label=r'$\alpha$', valmin=args.alphamin, valmax=args.alphamax, valinit=alpha)
            alpha_slider.on_changed(update_alpha)


            if args.use_noise_fbk:
                ## TODO rethink this part of the impl., maybe rename noise_scale to reflect new direction.
                ## initially i was planning on noise_scale being a feedback signal that
                ## adds noise for misprediction. now think the ga select will cover this,
                ## so noise_scale more properly named peak_noise_scale or something.
                ## but of course not crystalized on how adding noise to system. this
                ## sinusoidal noising process is something i feel like i want, i think its
                ## really a different noise scale than the noise that will drive the ga_selection

                def update_noise_scale(x):
                    nonlocal noise_scale
                    noise_scale = x
                noise_fbk_slider = Slider(fig.add_axes((0.2, 0.06, 0.65, 0.03)), label='Noise scale', valmin=args.noise_fbk_min, valmax=args.noise_fbk_max, valinit=noise_scale)
                noise_fbk_slider.on_changed(update_noise_scale)


            if args.use_ga:
                valid_topk_values = np.array(divisors(min(args.ga_topk_max, args.batch_dim)))

                def update_ga_sel_period(x):
                    nonlocal ga_sel_period
                    ga_sel_period = x
                ga_sel_period_slider = Slider(fig.add_axes((0.2, 0.09, 0.65, 0.03)), label=r'GA selection period', valmin=args.ga_sel_period_min, valmax=args.ga_sel_period_max, valinit=ga_sel_period, valstep=1)
                ga_sel_period_slider.on_changed(update_ga_sel_period)

                def update_ga_noise_scale(x):
                    nonlocal ga_noise_scale
                    ga_noise_scale = x
                ga_noise_scale_slider = Slider(fig.add_axes((0.2, 0.12, 0.65, 0.03)), label=r'GA noise scale', valmin=args.ga_noise_scale_min, valmax=args.ga_noise_scale_max, valinit=ga_noise_scale)
                ga_noise_scale_slider.on_changed(update_ga_noise_scale)

                def update_ga_topk(x):
                    nonlocal ga_topk
                    ga_topk = valid_topk_values[np.argmin((valid_topk_values - x) ** 2)] # get nearest valid topk value
                    if ga_topk_slider.val != ga_topk:
                        ga_topk_slider.set_val(ga_topk) # snap slider to valid topk value

                ga_topk_slider = Slider(fig.add_axes((0.2, 0.15, 0.65, 0.03)),
                                        label=r'GA top k to keep',
                                        valmin=args.ga_topk_min,
                                        valmax=args.ga_topk_max,
                                        valinit=ga_topk)
                ga_topk_slider.on_changed(update_ga_topk)

            plt.show()

        finally:
            if cap is not None:
                cap.release()
                cv2.destroyAllWindows()
                cv2.waitKey(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_nodes', '-N', required=True, type=int) ## TODO later, based on thoughts i wrote on paper, would like to not specify N like this, but rather go back specifying n^k and working recursively across scales. but for now I am going to work with this impl.
    parser.add_argument('--height', '-H', required=True, type=int)
    parser.add_argument('--width', '-W', required=True, type=int)
    parser.add_argument('--batch_dim', '-B', required=True, type=int)
    parser.add_argument('--channels', '-C', required=True, type=int)

    ## not using these for now.. instead just updating default args in CAN.__init__
    ## parser.add_argument('--wei_idx', required=False, default=None, type=int)
    ## parser.add_argument('--ker_idx', required=False, default=None, type=int)
    ## parser.add_argument('--stdin_idx', required=False, default=None, type=int)
    ## parser.add_argument('--stdout_idx', required=False, default=None, type=int)
    parser.add_argument('--key_idx', required=False, default=None, type=int)

    parser.add_argument('--clip_min', '-cmi', default=-2, type=float) # deprecated if using different activation .
    parser.add_argument('--clip_max', '-cma', default=2, type=float)

    parser.add_argument('--video_input', action='store_true')

    parser.add_argument('--complex', action='store_true')

    parser.add_argument('--sample_period', '-sp', default=1, type=int)
    parser.add_argument('--clean_period', required=False, default=0, type=float)

    parser.add_argument('--alphamin', required=False, default='0', type=float)
    parser.add_argument('--alphamax', required=False, default='1', type=float)

    parser.add_argument('--noise_fbk_min', required=False, default=None, type=float)
    parser.add_argument('--noise_fbk_max', required=False, default=None, type=float)

    ## parser.add_argument('--use_ga', action='store_true')
    ## parser.add_argument('--ga_sel_period', required=False, default=10, type=int)
    ## parser.add_argument('--ga_noise_scale', required=False, default=1, type=float)
    ## parser.add_argument('--ga_topk', required=False, default=1, type=int)


    parser.add_argument('--use_ga', action='store_true')

    parser.add_argument('--ga_sel_period_min', required=False, default=1, type=int)
    parser.add_argument('--ga_sel_period_max', required=False, default=100, type=int)

    parser.add_argument('--ga_noise_scale_min', required=False, default=0, type=float)
    parser.add_argument('--ga_noise_scale_max', required=False, default=1, type=float)

    parser.add_argument('--ga_topk_min', required=False, default=1, type=int)
    parser.add_argument('--ga_topk_max', required=False, default=None, type=int)


    ## TODO
    ## parser.add_argument('--activation')
    ## parser.add_argument('--step_implementation')



    args = parser.parse_args()

    # validate args

    ## TODO really probably shouldnt have args.sample_period and
    #  args.clean_period. the 'correct' way of doing this (i expect) is just
    #  having one. for now ignoring this though .. basically just forgot about
    #  sample period, hmm,  haha maybe want sample period to be controlled by
    #  some part of the state!
    ## TODO deal with this when implementing ga batch sel .. will deal with this when implementing 'decoding' .. this is next

    # TODO this assert is kind of a joke, since this is pretty uncharacteristic
    # of the rest of the code.. ah well will robustify throughout the code and
    # actually validate the args once ive reached a stable implementation
    assert args.ga_sel_period_min > 0, 'ga selection period must be positive'

    args.use_noise_fbk = (None not in [args.noise_fbk_min, args.noise_fbk_max])

    if args.ga_topk_max is None:
        args.ga_topk_max = args.batch_dim



    main(args)
