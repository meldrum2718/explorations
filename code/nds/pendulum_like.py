
import argparse

import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation


from .diffeq import DynSys, laplacian
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

            use_camera = 1
            freq = 1

            def step():
                nonlocal use_camera
                nonlocal freq

                t = 0

                inp = torch.zeros(H, W, C)

                while True:
                    t += 1
                    if use_camera:
                        _, inp = cap.read()
                        # convert inp to rgb with shape (h, w) and pixel values in [0, 1]
                        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
                        inp = cv2.resize(inp, (W, H), interpolation = cv2.INTER_AREA) / 255.0
                        inp = torch.Tensor(inp) # (H, W, C)
                        if C == 1:
                            inp = inp.mean(dim=-1).unsqueeze(-1) # make black and white
                        ds.step(inp, L=0)
                    else:

                        L = 0
                        cur_state = ds.state[L][-1]
                        dxdt = freq * -1 * cur_state.sin()
                        ds.step(dxdt, L=L+2)

                    yield t, ds

            ## set up figure for displaying the dyn.sys and some sliders
            nrows, ncols = get_appropriate_dims_for_ax_grid(K+1)
            fig, axs = plt.subplots(nrows, ncols)
            axs = axs.reshape(-1)
            for ax in axs: ax.axis('off')
            fig.subplots_adjust(bottom=0.25)
            cmap = 'grey' if C == 1 else None
            ims = [axs[k].imshow(activation(ds.state[k][-1]), cmap=cmap) for k in range(K + 1)]

            ## set up animation
            def draw_func(frame):
                t, can = frame
                for k, im in enumerate(ims):
                    x = activation(ds.state[k][-1])
                    im.set_data(x)
                    im.set_clim(x.min(), x.max())

                fig.suptitle(str(t))
                return ims

            ani = FuncAnimation(
                fig,
                func=draw_func,
                frames=step,
                interval=10,
                save_count=1,
            )


            def update_use_camera(x):
                nonlocal use_camera
                use_camera = int(x)
            use_camera_slider = Slider(fig.add_axes((0.2, 0.03, 0.65, 0.03)), label='Use Camera Input', valmin=0, valmax=1, valinit=1, valstep=1)
            use_camera_slider.on_changed(update_use_camera)

            def update_freq(x):
                nonlocal freq
                freq = x
            freq_slider = Slider(fig.add_axes((0.2, 0.06, 0.65, 0.03)), label='freq', valmin=0, valmax=1, valinit=freq)
            freq_slider.on_changed(update_freq)

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
