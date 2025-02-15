# from typing import Tuple
import torch
# import torch.nn as nn
import torch.nn.functional as F

from ..utils import inspect

class DynSys:
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

        self.state = [torch.zeros(K - L + 1, H, W, C) for L in range(K + 1)] # pyramid like

    def step(self, im, L=0):
        assert im.shape == (self.H, self.W, self.C)

        x = self.state[L].roll(shifts=(-1,), dims=(0,))
        x[-1] = im
        self.state[L] = x

        if L > 0:
            # propegate updards
            for k in range(L - 1, -1, -1): # [L-1, ..., 0]
                self.state[k] = self.state[k].roll(shifts=(-1,), dims=(0,))
                self.state[k][-1] = self.state[k][-2] + self.state[k+1][-1]

        if L < self.K:
            # propegate downwards
            for k in range(L + 1, self.K + 1): # [L+1, ..., K]
                self.state[k] = self.state[k].roll(shifts=(-1,), dims=(0,))
                self.state[k][-1] = self.state[k-1][-1] - self.state[k-1][-2]


Dx = torch.tensor([-1, 1]).reshape(1, 1, -1).to(torch.float32)
D2x = F.conv1d(Dx, Dx.flip(-1), padding=1).unsqueeze(0)
D2y = D2x.transpose(-1, -2)

def laplacian(im):
    assert im.ndim == 4, 'im should be a (B, C, H, W) tensor'
    return torch.sum(
        torch.stack((
            F.conv2d(im, D2x, padding='same'),
            F.conv2d(im, D2y, padding='same')),
            dim=0),
        dim=0)


def mesh2d(H, W, xmin=0, xmax=1, ymin=0, ymax=1, use_xy_format=False):
    i = torch.linspace(ymin, ymax, H).reshape(H, 1).expand(H, W)
    j = torch.linspace(xmin, xmax, W).reshape(1, W).expand(H, W)
    if use_xy_format:
        return torch.stack((j, i))
    return torch.stack((i, j), dim=-1)


def gauss_ker2d(nh, nw, xmin, xmax, ymin, ymax, mu, sigma):    
    mesh = mesh2d(nh, nw, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    sigma_inv = torch.inverse(sigma)
    
    flat_mesh = mesh.reshape(-1, 2)
    centered_mesh = flat_mesh - mu
    quad_form = torch.sum(centered_mesh @ sigma_inv * centered_mesh, dim=1)
    return  torch.exp(-0.5 * quad_form).reshape(nh, nw
                                               ) / (
                                                   2 * torch.pi * torch.linalg.det(sigma) ** 0.5)
