import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import inspect, normalize


is_first_bmm = True

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


    def output(self, node_idx=None):
        """ output the state ready for matplotlib."""
        if node_idx is None:
            out = self.state
        else:
            out = self.state[node_idx] ## TODO self.state

        if self.color:
            out = out[:, :, 0:3]  # just look at first 3 channels
        else:
            out = torch.mean(self.state[node_idx], dim=-1)# (n**k, n**k, C) -> (n**k, n**k) mean across channel dim to get b/w image
        
        out = out.detach().cpu()
        out = out.clip(0, 1)

        return out


    def input(self, inp, node_idx=None, L=2, alpha=1):
        """ update state[idx] as a convex combination with inp."""
        ## TODO take in a 'nest' param, then input into node_idx in the nested array
        if node_idx is None:
            node_idx = self.stdin_idx

        H, W, C = inp.shape
        N, K = self.N, self.K

        state = self.state.reshape(N**L, N**(K-L), N**L, N**(K-L), self.C)
        state = state.permute(0, 2, 1, 3, 4) # (n^l, n^l, n^(k-l), n^(k-l), c)
        state = state.reshape(-1, N**(K-L), N**(K-L), self.C) # B, n^(k-l), n^(k-l), c)
        state = state.reshape(-1, N, N**(K-L-1), N, N**(K-L-1), self.C) # (B, n, n^(k-l-1), n^(k-l-1), c)
        state = state.permute(0, 1, 3, 2, 4, 5) # (B, n, n, n^(k-l-1), n^^(k-l-1), c)
        state = state.reshape(-1, N**2, N**(K-L-1), N**(K-L-1), self.C)
        B = state.shape[0]
        inp = F.interpolate(
            inp.unsqueeze(0).permute(0, 3, 1, 2),
            size=(N**(K-L-1), N**(K-L-1)),
            mode='bilinear',
            antialias=True
        ).permute(0, 2, 3, 1)



        if self.C >= C: # if more channels in self.state than in inp
            state[:, node_idx, :, :, 0:C] = alpha * inp  +  (1 - alpha) * state[:, node_idx, :, :, 0:C]
        elif self.C == 1:
            inp = torch.mean(inp, dim=1) # (1, C, H, W) -> (1, 1, H, W)
            state[:, node_idx] = alpha * inp  +  (1 - alpha) * state[node_idx]
        else:
            raise Exception('TODO have not handled case where (self.C != 1) and (C > self.C)')

        self.state = state.reshape(
            B, N, N, N**(K-L-1), N**(K-L-1), self.C
        ).permute(0, 1, 3, 2, 4, 5
        ).reshape(N**L, N**L, N**(K-L), N**(K-L), self.C
        ).reshape(N**L, N**L, N**(K-L), N**(K-L), self.C
        ).permute(0, 2, 1, 3, 4
        ).reshape(self.state.shape)

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


    def step(self, alpha=0.01, Lmax=0):
        """
        update: X += alpha * dXdt

        ## TODO make alpha be something more interesting than just slider
        controlled probably.. lets let the network parameterize this i think.
        or potentially, when building more hand cratfted architectures, would
        presumably like to be able to control the 'inertia' (alpha) of the
        various nodes..


        Now:
            going to build a more interesting information flow into the step.
            would like to use a geometric graph type approach. now, something
            that seems the simplest is simply letting the context for the self
            attention be a sum of the neighboring nodes. yeah i think this is
            the way, this way can have many edges but still only need to to do
            one attn op per node.

            ok, so then, how to implement: could just do a sum over
            torch.roll(dims=(1,2)) copies of
            state:(B,n,n,n^(k-l-1),n&(k-l-1),c). observe that this introduces
            nested tori inside the 2d array.. could be nice, could also be
            confounding to the 2d structure of the hierarchical nesting.. (tori in the sense of the unit square with opposite edges identified)

            alternatively could do a sum over torch.roll(dims=(1,2)) copies of
            state:(n^k, n^k, n^(k-l), n^(k-l)), or even a batched matmul (hmm
            ideally sparse id think..) .. thinking about this and it seems
            really more of a correct way .. idk feels like a nice way of
            avoiding hard edges in the state, and then can try having input
            cover the whole state and then ensure that there are enough
            channels for the diffeq to play out in the hidden channels. then
            will like to visualize the hidden channels as well, so this
            requires changing main.py
            the batched matmul would allow for encoding arbitrary graphs as the
            connectivity structure .. then could even have something like
            different adjacency structures for different scales .. could open
            up some interesting possibilities..


            requiring least changes is just renaming letting context be a sum
            of torch.rolls of the current state, then doing cross attention on
            (state, context). or perhaps a torch.conv with a gaussian kernel.
            probably this is the best way to try for right now. either of these
            two things are what i ought to implement next ..

            ok, plan: lets try the local sum because it seems so very quick to
            implement. then after that lets make state.shape be the more
            sensible (n^k,n^k,n^(k-l),n^(k-l), c), and then we can just do
            convolutions by flattening last 3 dims, and dont really need to
            worry about off by 1 errors so much. might also want to change the
            off by one errors found in `def input``
        """


        # attn
        # inspect('self.state', self.state)
        N, K, C = self.N, self.K, self.C
        dxdt = torch.zeros_like(self.state)

        L = torch.randint(0, Lmax+1, size=(1,)).item()

        # pass to L-nested view
        state = self.state.reshape(N**L, N**(K-L), N**L, N**(K-L), C)
        state = state.permute(0, 2, 1, 3, 4) # (n^l, n^l, n^(k-l), n^(k-l), c)
        state = state.reshape(-1, N**(K-L), N**(K-L), C) # B, n^(k-l), n^(k-l), c)
        state = state.reshape(-1, N, N**(K-L-1), N, N**(K-L-1), C) # (B, n, n^(k-l-1), n^(k-l-1), c)
        state = state.permute(0, 1, 3, 2, 4, 5) # (B, n, n, n^(k-l-1), n^(k-l-1), c)
        B = state.shape[0]

        # inspect('state', state)

        shifts = torch.randint(-1, 2, size=(2,)).tolist()
        nei = state.roll(shifts=shifts, dims=(1, 2))

        state = state.reshape(-1, N**2, N**(K-L-1), N**(K-L-1), C)

        query = self.ker(state, self.query_idx)
        key = self.ker(state, self.key_idx)
        value = self.ker(state, self.value_idx)

        # inspect('query:', query)
        # inspect('key:', key)
        # inspect('value:', value)

        # query = query.reshape(B, N**2, -1) ## TODO changed from reshape(B,N,-1) to (B,N**2,-1), seems a bit more interesting.. TODO reason about exactly what were doing with the attn operation..
        # key = key.reshape(B, N**2, -1)
        # value = value.reshape(B, N**2, -1)

        query = state.reshape(B, N**2, -1)
        key = value = nei.reshape(B, N**2, -1)

        # scale_factor = 1 / math.sqrt(query.size(-1))

        attn = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            # scale=scale_factor,
        ).reshape(state.shape)

        # inspect('attn', attn)

        dxdt = torch.zeros_like(state)
        dxdt += attn

        # pass back to flat view
        dxdt = dxdt.reshape(
            B, N, N, N**(K-L-1), N**(K-L-1), C
        ).permute(0, 1, 3, 2, 4, 5
        ).reshape(N**L, N**L, N**(K-L), N**(K-L), C
        ).reshape(N**L, N**L, N**(K-L), N**(K-L), C
        ).permute(0, 2, 1, 3, 4
        ).reshape(self.state.shape)

        self.state += alpha * dxdt
        self.proj_to_torus()
