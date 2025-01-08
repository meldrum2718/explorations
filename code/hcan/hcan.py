import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import inspect, normalize


# def bmm(x, w, verbose):
#     """ assuming x, w: (N, B, C, H, W), perform batch matmul."""
#     if verbose:
#         inspect('x', x)
#         inspect('w', w)
#     x = x.transpose(0, 1)
#     B, N, C, H, W = x.shape
#     x = x.reshape(B, N, -1)
#     return torch.bmm(
#         w.transpose(-1, -2),
#         x
#     ).reshape(B, N, C, H, W).transpose(0, 1)


class HCAN:
    """ A pde on a tensor state with a hierarchical attention coupling."""
    def __init__(
        self,
        N: int,
        K: int,
        C: int,
        wei_idx: Tuple[int, int] = 0,
        query_idx: Tuple[int, int] = None,
        key_idx: Tuple[int, int] = 1,
        value_idx: Tuple[int, int] = 1,
        stdin_idx: Tuple[int, int] = 2,
        stdout_idx: Tuple[int, int] = 3,
    ):

        self.N = N
        self.K = K
        self.C = C

        self.rand = torch.randn
        self.rand_like = torch.randn_like

        self.state = self.rand(N**K, N**K, C, dtype=torch.float32)
        self.proj_to_torus()

        self.wei_idx = wei_idx
        self.query_idx = query_idx
        self.key_idx = key_idx
        self.value_idx = value_idx
        self.stdin_idx = stdin_idx
        self.stdout_idx = stdout_idx

        self.color = (C >= 3)


    #def input(self, inp, node_idx=None, L=0, alpha=1):
    #    """ update state[idx] as a convex combination with inp."""

    #    N, K, C = self.N, self.K, self.C

    #    state = self.state.reshape(N**L, N**(K-L), N**L, N**(K-L), C)
    #    state = state.permute(0, 2, 1, 3, 4) # (N^L, N^L, N^(K-L), N^(K-L), C)
    #    state = state.reshape(N**(2*L), N**(K-L), N**(K-L), C) # N^(2*L), N^(K-L), N^(K-L), C)

    #    inp = F.interpolate(
    #        inp.permute(2, 0, 1).unsqueeze(0),
    #        size=(N**(K-L), N**(K-L)),
    #        mode='bilinear',
    #        antialias=True
    #    ).permute(0, 2, 3, 1).squeeze(-1) # (1, N**(K-L), N**(K-L))

    #    if node_idx is not None:
    #        state[node_idx, :, :, L] = alpha * inp  +  (1 - alpha) * state[node_idx, :, :, L] # put input into channel layer L
    #    else:
    #        state[..., L] = alpha * inp + (1 - alpha) * state[..., L]

    #    state = state.reshape(
    #        N**L, N**L, N**(K-L), N**(K-L), C
    #    ).permute(0, 2, 1, 3, 4 # (N**L, N**(K-L), N**L, N**(K-L), C)
    #    ).reshape(N**K, N**K, C)

    #    self.state = state
    #    self.proj_to_torus()


    def input(self, inp, node_idx=None, L=0, alpha=1):
        """ update state[idx] as a convex combination with inp."""

        C_inp = inp.shape[2]
        N, K, C = self.N, self.K, self.C

        state = self.state.reshape(N**L, N**(K-L), N**L, N**(K-L), C)
        state = state.permute(0, 2, 1, 3, 4) # (N^L, N^L, N^(K-L), N^(K-L), C)
        state = state.reshape(N**(2*L), N**(K-L), N**(K-L), C) # N^(2*L), N^(K-L), N^(K-L), C)

        inp = F.interpolate(
            inp.permute(2, 0, 1).unsqueeze(0),
            size=(N**(K-L), N**(K-L)),
            mode='bilinear',
            antialias=True
        ).permute(0, 2, 3, 1) # (1, N**(K-L), N**(K-L), C_inp)


        if node_idx is not None:
            state[node_idx, :, :, 0:C_inp] = alpha * inp  +  (1 - alpha) * state[node_idx, :, :, 0:C_inp]
        else:
            state[..., 0:C_inp] = alpha * inp + (1 - alpha) * state[..., 0:C_inp]

        state = state.reshape(
            N**L, N**L, N**(K-L), N**(K-L), C
        ).permute(0, 2, 1, 3, 4 # (N**L, N**(K-L), N**L, N**(K-L), C)
        ).reshape(N**K, N**K, C)

        self.state = state
        self.proj_to_torus()





    def add_noise(self, noise_scale: float):
        self.state += noise_scale * self.rand_like(self.state)
        self.proj_to_torus()


    def wei(self):
        wei = self.state[self.wei_idx] # (B, C, H, W)
        wei = torch.mean(wei, dim=1).unsqueeze(1) # (B, C, H, W) -> (B, 1, H, W)
        wei = F.interpolate(wei, size=(self.N, self.N)).squeeze(1) # (B, H, W) -> (B, N, N)
        return wei

    def nest(self, x, L):
        """
            Args:
                L: int
                x: (N**K, N**K, C)
            Returns:
                nested: (N**L, N**L, N**(K-L), N**(K-L), C) with nested[i, j] =
        """
        N, K, C = self.N, self.K, self.C
        return x.view(
            N**L, N**(K-L), N**L, N**(K-L), C
        ).permute(0, 2, 1, 3, 4) # (N**L, N**L, N**(K-L), N**(K-L), C)

    def flatten(self, x, L):
        """ 
            Args:
                L: int
                x: (N**L, N**L, N**(K-L), N**(K-L), C)
            Returns:
                flattened: (N**K, N**K, C)
        """
        N, K, C = self.N, self.K, self.C
        return x.permute(0, 2, 1, 3, 4).reshape(N**K, N**K, C)


    def ker(self, x, control_node_idx):
        """
        Use control node index to read a matching between nodes.
        Args:
            x: (B, N, H, W, C)
        Returns:
            ker: (B, N, H, W, C)
        """
        B, N, H, W, C = x.shape
        if control_node_idx is None:
            return x
        k = x[:, control_node_idx] # (B, C, H, W)                                        ## get the node that parameterizes the kernels
        #inspect('k1', k)
        k = torch.mean(k, dim=-1).unsqueeze(1) # (B, H, W, C) -> (B, 1, H, W)             ## make into a b/w image
        #inspect('k2', k)
        k = F.interpolate(k, size=(N, N), mode='bilinear', antialias=True).squeeze(1) # (B, H, W) -> (B, N, N)  ## make correct shape
        #inspect('k3', k)
        k = torch.argmax(k, dim=1) # (B, N, N) -> (B, N)  ## now get indices             ## obtain batched list of indices, N indices per batch
        #inspect('k4', k)
        batch_idxs = torch.arange(B).unsqueeze(-1).expand(B, N) # (B, N)  ## generate batch indices
        #inspect('bi', batch_idxs)
        ker = x[batch_idxs, k] # (B, N, H, W, C)                                         ## index into state to get a kernel for each node, for each batch dim
        #inspect('k5', ker)
        return ker


    def proj_to_torus(self):
        self.state = torch.frac(1 + torch.frac(self.state))


    def step(self, alpha=0.01, L=0, C_min=0, C_max=None):
        """
        update: X += alpha * dXdt


        """

        N, K, C = self.N, self.K, self.C

        # pass to L-nested view
        state = self.state
        state = state.reshape(N**L, N**(K-L), N**L, N**(K-L), C)
        state = state.permute(0, 2, 1, 3, 4) # (N^L, N^L, N^(K-L), N^(K-L), C)

        ## TODO instead of this random (dx, dy) shift, prefer to take some 3x3 matrix and warp a mesh over R^2 like in image warping i.e. warp via a homography.
        shifts = torch.randint(low=-N**L, high=N**L+1, size=(2,)).tolist()

        nei = state.roll(
            shifts=shifts, dims=(0, 1)
        ).reshape(1, N**(2*L), C*N**(2*(K-L)))

        state = state.reshape(1, N**(2*L), C*N**(2*(K-L)))

        # nei_interp = F.interpolate(
        #     nei.permute(0, 3, 1, 2),
        #     size=(N, N),
        #     mode='bilinear',
        #     antialias=True,
        # ).permute(0, 2, 3, 1) # (N**(2*L), N, N, C)

        # state_interp = F.interpolate(
        #     state.permute(0, 3, 1, 2),
        #     size=(N, N),
        #     mode='bilinear',
        #     antialias=True,
        # ).permute(0, 2, 3, 1) # (N**(2*L), N, N, C)

        # query = state.reshape(N**(2*L), N**2, -1)
        # key = value = nei.reshape(N**(2*L), N**2, -1)

        dxdt = F.scaled_dot_product_attention(
            query=state,
            key=nei,
            value=nei,
        ) # (1, N**(2*L), C*N**(2*(K-L)))

        dxdt = dxdt.reshape(N**L, N**L, N**(K-L), N**(K-L), C
        ).permute(0, 2, 1, 3, 4 # (N**L, N**(K-L), N**L, N**(K-L))
        ).reshape(N**K, N**K, C)

        # dxdt = F.interpolate(
        #     dxdt.permute(0, 3, 1, 2),
        #     size=(N**(K-L), N**(K-L)),
        #     mode='bilinear',
        #     antialias=True,
        # ).permute(0, 2, 3, 1) # (N**(2*L), N**(K-L), N**(K-L), C) 

        # dxdt = dxdt.reshape(
        #     N**L, N**L, N**(K-L), N**(K-L), C
        # ).permute(0, 2, 1, 3, 4 # (N**L, N**(K-L), N**L, N**(K-L), C)
        # ).reshape(N**K, N**K, C)

        self.state[..., C_min:C_max] += alpha * dxdt[..., C_min:C_max] # only worry about C_min:C_max channels
        self.proj_to_torus()
