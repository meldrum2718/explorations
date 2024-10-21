import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from rtn import RTN
from utils import get_video_capture, rgb2grey, normalize


def sample_prod_params(X, n, k, color=False):
    """ Return parameters for a product of X with a subtensor of itself.

        Just going to do a really simple sampling for now. future, perhaps
        make it be more built/structured, like with a conv stem or so. For now
        it is going to just be directly computed from reading permutations and
        indices from submatrices of X in the top left-ish vicinity of X.

        Returns:
            (pre_xlp, pre_ksp, pre_ki, pre_klp), X_latent_perm, ker

        TODO think about instead returning ksp, knd, ki, klp. q: is this a nice general representation of the parameters into the product?. a: it seems nice for products between X and a subtensor of X.

    """
    if color:
        X_ndim = 2 * k + 1
    else:
        X_ndim = 2 * k

    nested_shape = (n for _ in range(X_ndim))
    
    if color:
        X_color = X
        X = rgb2grey(X)
    X_latent_perm =   np.argsort(pre_xlp := np.sum(X[0:n, 0:X_ndim], axis=0))
    ker_sample_perm = np.argsort(pre_ksp := np.sum(X[2*n:3*n, 0:X_ndim], axis=0))
    ker_ndim =         np.argmax(pre_knd := np.sum(X[n:2*n, 0:k-2], axis=0)) + 2 # ker_ndim in [2, 3, ..., k]
    ker_indices = tuple(np.argmax(pre_ki := X[3*n:4*n, 0:X_ndim-ker_ndim], axis=0))
    ker_latent_perm = np.argsort(pre_klp := np.sum(X[4*n:5*n, 0:ker_ndim], axis=0))


    # print('x.s', X.shape, 'x.numel', X.size)
    # print('nested shape', nested_shape, 'ns.size')
    if color:
        X = X_color

    ker = X.reshape(*nested_shape)
    ker = ker.transpose(*ker_sample_perm)
    ker = ker[ker_indices]
    ker = ker.transpose(*ker_latent_perm)


    # print(X_latent_perm.shape)
    # print(ker.shape)


    return (pre_xlp, pre_ksp, pre_ki, pre_klp), X_latent_perm, ker


def main(args):
    cap = None
    try:
        n, k = args.n, args.k

        if args.video_input:
            cap = get_video_capture()

        if None in (args.h, args.w):
            args.h = args.w = n**(k-1)

        rtn = RTN(n, k, color=args.color, clip_min=args.clip_min, clip_max=args.clip_max)

        fig = plt.figure(figsize=(14, 7))
        im_ax = fig.add_subplot(1, 2, 1)
        pxlp_ax = fig.add_subplot(4, 2, 2)
        pksp_ax = fig.add_subplot(4, 2, 4)
        pki_ax = fig.add_subplot(4, 2, 6)
        pklp_ax = fig.add_subplot(4, 2, 8)

        im = im_ax.imshow(normalize(rtn.X), cmap='grey')

        def frame_gen():
            t = 0
            while True:
                t += 1
                inp = None
                if (args.sample_period is not None) and (t % args.sample_period == 0) and (cap is not None):
                    _, inp = cap.read()
                    # convert inp to rgb with shape (h, w) and pixel values in [0, 1]
                    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
                    inp = cv2.resize(inp, (args.w, args.h), interpolation = cv2.INTER_AREA) / 255.0
                    if not args.color:
                        inp = rgb2grey(inp)

                pre, xlp, ker = sample_prod_params(rtn.X, rtn.n, rtn.k, args.color)
                state = rtn.step(latent_perm=xlp, ker=ker, inp=inp, alpha=args.alpha).copy()
                yield t, (pre, ker, state)

        def draw_func(frame):
            t, frame = frame
            pre, ker, state = frame
            pre_xlp, pre_ksp, pre_ki, pre_klp = pre
            im.set_data(normalize(state))
            pxlp_ax.cla(); pxlp_ax.stem(pre_xlp); pxlp_ax.set_title("X latent permutation")
            pksp_ax.cla(); pksp_ax.stem(pre_ksp); pksp_ax.set_title("Kernel sample permutation")
            pki_ax.cla(); pki_ax.imshow(pre_ki); pki_ax.set_title("Kernel")
            pklp_ax.cla(); pklp_ax.stem(pre_klp); pklp_ax.set_title("Kernel latent permutation.")
            fig.suptitle(str(t))
            return im

        ani = FuncAnimation(
            fig,
            func=draw_func,
            frames=frame_gen,
            interval=50,
            save_count=0,
        )

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
    parser.add_argument('--alpha', required=False, default='0.01', type=float)
    parser.add_argument('--video_input', action='store_true')
    parser.add_argument('--color', '-c', action='store_true')
    parser.add_argument('--h', default=None, type=int)
    parser.add_argument('--w', default=None, type=int)
    parser.add_argument('--clip_min', '-cmi', default=-2, type=float)
    parser.add_argument('--clip_max', '-cma', default=2, type=float)
    parser.add_argument('--sample_period', '-sp', default=None, type=int)


    args = parser.parse_args()

    if args.sample_period is not None: args.video_input = True
    if args.color: args.video_input = True

    main(args)
