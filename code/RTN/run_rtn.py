import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

from .rtn import RTN
from ..utils import get_video_capture, rgb2grey, normalize


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


        cmap = None if args.color else 'grey'
        im = im_ax.imshow(normalize(rtn.output()[0]), cmap=cmap)

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
                    inp = np.stack([inp] * args.batch_dim, axis=0)
                    assert np.all(inp.shape == rtn.output().shape), f'inp shape: {inp.shape},   rtn.out.shape: {rtn.output().shape}'

                if inp is not None:
                    ker = inp
                else:
                    ker = rtn.X.reshape(rtn.flat_shape)
                state = rtn.step(ker=ker, C=rtn.X.reshape(rtn.flat_shape), inp=inp, alpha=alpha, noise_fbk=noise_fbk)

                yield t, state

        def draw_func(frame):
            t, state = frame
            state = state[0] # for now, just look at first row in batch dim
            im.set_data(normalize(state))
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
    parser.add_argument('--batch_dim', '-bd', default=1, type=int)


    args = parser.parse_args()

    # validate args
    if args.sample_period:
        args.video_input = True
    if args.video_input:
        args.sample_period = args.sample_period or 1

    if args.color:
        assert args.n == 3, 'handling color for n != 3 not currently supported.'

    main(args)
