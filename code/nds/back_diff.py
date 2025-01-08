# from typing import Tuple
import torch
# import torch.nn as nn
# import torch.nn.functional as F

from ..utils import inspect

class KthOrderBackDiff:
    def __init__(
        self,
        H: int,
        W: int,
        C: int,
        K: int,
    ):

        self.H = H
        self.W = W
        self.C = C
        self.K = K

        self.state = [torch.zeros(K + 1 - k, H, W, C) for k in range(K + 1)] # pyramid like

    def step(self, im):
        assert im.shape == (self.H, self.W, self.C)

        x = self.state[0].roll(shifts=(-1,), dims=(0,))
        x[-1] = im
        self.state[0] = x

        for k in range(1, self.K + 1):
            Dk = self.state[k].roll(shifts=(-1,), dims=(0,))
            Dk[-1] = self.state[k-1][-1] - self.state[k-1][-2]
            self.state[k] = Dk
