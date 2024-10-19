import argparse

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from rtn import RTN

# import torch
# import torch.nn.functional as F



def main(args):
    n, k = args.n, args.k
    rtn = RTN(n, k)

    fig, ax = plt.subplots()
    im = plt.imshow(rtn.X)

    def frame_gen():
        t = 0
        while True:
            t += 1
            yield t, rtn.step(alpha=args.alpha).copy()

    def draw_func(frame):
        t, frame = frame
        im.set_data(frame)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', required=True, type=int)
    parser.add_argument('--k', required=True, type=int)
    parser.add_argument('--alpha', required=False, default='0.01', type=float)
    args = parser.parse_args()

    main(args)



## code for running live video i/o feeding into the ips/tbn/nca/rtn
## import cv2
## 
## def main():
##     try:
##         cap = cv2.VideoCapture(0)
##         if not cap.isOpened():
##             print('cannot open camera')
##             return
## 
##         fig, axs = plt.subplots(axs_nrows, axs_ncols, figsize=(9, 9))
##         axs = axs.reshape(-1)
##         ims = []
##         init_frame = nca.state.detach().squeeze(0).permute(1, 2, 3, 0)
## 
## 
##         for di in range(d):
##             axs[di].set_xticks([])
##             axs[di].set_yticks([])
##             axs[di].set_title(di)
##             im = axs[di].imshow(init_frame[di], cmap='viridis')
##             ims.append(im)
## 
##         def step():
##             t = 0
##             while True:
##                 t += 1
##                 ret, img = cap.read()
##                 img = cv2.resize(img, (h, w)) / 255.0
##                 inp = torch.Tensor(img).permute(2, 0, 1).reshape(1, 3, 1, h, w)
##                 nca.step(inp)
##                 yield t, nca.state.detach().squeeze(0).permute(1, 2, 3, 0)
## 
##         def draw_func(frame):
##             t, frame = frame
##             for di in range(d):
##                 ims[di].set_data(frame[di])
##             fig.suptitle(str(t))
##             return ims
## 
##         ani = FuncAnimation(fig, draw_func, step, interval=50, save_count=10)
##         plt.tight_layout()
##         plt.show()
## 
##     finally:
##         cap.release()
##         cv2.destroyAllWindows()
##         cv2.waitKey(1)
##         plt.cla()
## 
## 
## 
## 
## if __name__ == '__main__':
##     main()
