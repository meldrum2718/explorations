import argparse
# import tracemalloc

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

from rtn import RTN
from utils import get_video_capture, rgb2grey, normalize


def main(args):
    cap = None
    try:
        n, k = args.n, args.k

        if args.video_input:
            cap = get_video_capture()

        h = w = n**k

        rtn = RTN(n, k, batch_dim=args.batch_dim, color=args.color, clip_min=args.clip_min, clip_max=args.clip_max)

        fig = plt.figure(figsize=(14, 7))
        im_ax = fig.add_subplot()

        im = im_ax.imshow(normalize(rtn.output()), cmap='grey')

        alpha = 0
        noise_fbk = 0

        def frame_gen():
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
                    if not args.color:
                        inp = rgb2grey(inp)

                ker = rtn.X if inp is None else inp
                state = rtn.step(ker=ker, C=rtn.X.reshape(rtn.flat_shape), inp=inp, alpha=alpha, noise_fbk=noise_fbk)

                yield t, state

        def draw_func(frame):
            t, state = frame
            im.set_data(normalize(state))
            # im.axes.figure.canvas.draw_idle()
            fig.suptitle(str(t))
            return [im]

        ani = FuncAnimation(
            fig,
            func=draw_func,
            frames=frame_gen,
            interval=10,
            save_count=1,
        )

        def update_alpha(x):
            nonlocal alpha
            alpha = x
        alpha_slider = Slider(fig.add_axes([0.2, 0.05, 0.65, 0.03]), label=r'$\alpha$', valmin=args.alphamin, valmax=args.alphamax, valinit=alpha)
        alpha_slider.on_changed(update_alpha)

        def update_noise_fbk(x):
            nonlocal noise_fbk
            noise_fbk = x
        noise_fbk_slider = Slider(fig.add_axes([0.2, 0.10, 0.65, 0.03]), label='Noise feedback', valmin=args.noise_fbk_min, valmax=args.noise_fbk_max, valinit=noise_fbk)
        noise_fbk_slider.on_changed(update_noise_fbk)

        plt.show()

    except Exception as e:
        print(e)
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
    parser.add_argument('--video_input', action='store_true')
    parser.add_argument('--color', '-c', action='store_true')
    parser.add_argument('--clip_min', '-cmi', default=-2, type=float)
    parser.add_argument('--clip_max', '-cma', default=2, type=float)
    parser.add_argument('--sample_period', '-sp', default=None, type=int)
    parser.add_argument('--alphamin', required=False, default='0', type=float)
    parser.add_argument('--alphamax', required=False, default='1', type=float)
    parser.add_argument('--noise_fbk_min', required=False, default='0', type=float)
    parser.add_argument('--noise_fbk_max', required=False, default='0.01', type=float)
    parser.add_argument('--batch_dim', '-bd', default=0, type=int)



    args = parser.parse_args()

    # validate args
    if args.sample_period:
        args.video_input = True
    if args.video_input:
        args.sample_period = args.sample_period or 1

    if args.color:
        assert args.n >= 3, 'handling color for n < 3 not supported.'

    main(args)






















####### deprecated ###########

# def sample_prod_params(X, n, k, batch_dim=None, color=False):
#     """ Return parameters for a product of X with a subtensor of itself.
# 
#         Just going to do a really simple sampling for now. future, perhaps
#         make it be more built/structured, like with a conv stem or so. For now
#         it is going to just be directly computed from reading permutations and
#         indices from submatrices of X in the top left-ish vicinity of X.
# 
#         Returns:
#             (pre_xlp, pre_ksp, pre_ki, pre_klp), X_latent_perm, ker
# 
#         TODO think about instead returning ksp, knd, ki, klp. q: is this a nice general representation of the parameters into the product?. a: it seems nice for products between X and a subtensor of X.
# 
#     """
#     
#     if batch_dim is not None:
#         X = X.reshape(n**batch_dim, n**k, n**batch_dim, n**k).reshape(0, 2, 1, 3)
# 
#     if color:
#         X_ndim = 2 * k + 1
#     else:
#         X_ndim = 2 * k
# 
#     nested_shape = (n for _ in range(X_ndim))
#     
#     if color:
#         X_color = X
#         X = rgb2grey(X)
#     X_latent_perm =   np.argsort(pre_xlp := np.sum(X[0:n, 0:X_ndim], axis=0))
#     ker_sample_perm = np.argsort(pre_ksp := np.sum(X[2*n:3*n, 0:X_ndim], axis=0))
#     ker_ndim =         np.argmax(pre_knd := np.sum(X[n:2*n, 0:k-2], axis=0)) + 2 # ker_ndim in [2, 3, ..., k]
#     ker_indices = tuple(np.argmax(pre_ki := X[3*n:4*n, 0:X_ndim-ker_ndim], axis=0))
#     ker_latent_perm = np.argsort(pre_klp := np.sum(X[4*n:5*n, 0:ker_ndim], axis=0))
# 
# 
#     # print('x.s', X.shape, 'x.numel', X.size)
#     # print('nested shape', nested_shape, 'ns.size')
#     if color:
#         X = X_color
# 
#     ker = X.reshape(*nested_shape)
#     ker = ker.transpose(*ker_sample_perm)
#     ker = ker[ker_indices]
#     ker = ker.transpose(*ker_latent_perm)
# 
#     return (pre_xlp, pre_ksp, pre_ki, pre_klp), X_latent_perm, ker
