"""
A little script to look at the decision boundary of a toy neural network.
"""



import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.neural_network import MLPClassifier


def nn(xi, xj, theta, activation, classify=True):
    """
    A tiny toy neural network for binary classification.

    Args:
        x: np.ndarray with shape (B, 2)
        theta: np.ndarray with shape (6,)
    Out:
        preds: np.ndarray with shap (B, 1)
    """
    x1 = activation(theta[0] * xi + theta[1] * xj)
    x2 = activation(theta[2] * xi + theta[3] * xj)
    x  = activation(theta[4] * x1 + theta[5] * x2)

    if classify:
        x = np.round(x).astype(int)
    return x


def sigmoid(x):
    return  1 / (1 + np.exp(-x))


def main(args):
    fig = plt.figure()
    ax = fig.add_axes([0.2, 0.30, 0.65, 0.65])
    ax.axis('off')

    xi, xj = np.meshgrid(np.linspace(args.xmin, args.xmax, args.n), np.linspace(args.ymin, args.ymax, args.n))
    xi = xi.ravel()
    xj = xj.ravel()

    sliders = [
        Slider(fig.add_axes([0.2, 0.08, 0.65, 0.03]), label = r'$\theta_0$', valmin=args.valmin, valmax=args.valmax, valinit=0),
        Slider(fig.add_axes([0.2, 0.11, 0.65, 0.03]), label = r'$\theta_1$', valmin=args.valmin, valmax=args.valmax, valinit=0),
        Slider(fig.add_axes([0.2, 0.14, 0.65, 0.03]), label = r'$\theta_2$', valmin=args.valmin, valmax=args.valmax, valinit=0),
        Slider(fig.add_axes([0.2, 0.17, 0.65, 0.03]), label = r'$\theta_3$', valmin=args.valmin, valmax=args.valmax, valinit=0),
        Slider(fig.add_axes([0.2, 0.20, 0.65, 0.03]), label = r'$\theta_4$', valmin=args.valmin, valmax=args.valmax, valinit=0),
        Slider(fig.add_axes([0.2, 0.23, 0.65, 0.03]), label = r'$\theta_5$', valmin=args.valmin, valmax=args.valmax, valinit=0),
    ]



    def update(val):
        theta = np.array([slider.val for slider in sliders])
        preds = nn(xi, xj, theta, sigmoid, classify=args.classify)

        ax.cla()
        ax.imshow(preds.reshape(args.n, args.n), cmap='grey')
        ax.axis('off')

    for slider in sliders:
        slider.on_changed(update)

    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h', type=int, default=-1)
    parser.add_argument('--w', type=int, default=-1)
    parser.add_argument('--valmin', '-min', type=float, default=-5, help='slider min value')
    parser.add_argument('--valmax', '-max', type=float, default=5, help='slider max value')

    parser.add_argument('--xmax', type=float, default=10)
    parser.add_argument('--ymax', type=float, default=10)

    parser.add_argument('--xmin', type=float, default=-10)
    parser.add_argument('--ymin', type=float, default=-10)


    parser.add_argument('--n', type=float, default=100)
    parser.add_argument('--classify', '-c', action='store_true')

    args = parser.parse_args()
    main(args)

