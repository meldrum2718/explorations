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


    def step(self, alpha=0.01, L=0):
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
            state:(B,n,n,n^(k-l-1),n^(k-l-1),c). observe that this introduces
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


            12/27 thinking about it .. ive just implemented information flow
            across random translations of the grid at a hierarchical scale.
            gives interesting results, even with doing it a bit wrong (the
            nested tori inside the 2d array do not appear to help information
            flow (really this should be expected as the nested tori flows in
            the (n^k,n^k, n,n,n^(k-l-1),n^(k-l-1)) approach induces arbitrary
            unhealthy boundaries as we view at different scales).
            now thinking: precisely what i am doing here is thinking of the
            information flow graph as being a caley graph, and i am randomly
            choosing a color and then flowing cross attention along the edges
            of the chosen color.
            question: how to do the sort oh recursive state breakdown in more exotic groups than Z_n^2, in particular thinking of lie groups probably
                think we want a sort of mesh over the space, assign a vector to
                each node in the mesh, then view mesh with a recursive partitioning i.e. a kd-tree like.
                thought about it a bit: seems to me like this:
                    want to deal with a lie group. by definition this is a
                    differentiable manifold, i.e. locally is a piece of R^d. Ok
                    then we just have the nodes be a mesh over R^d, and do the
                    recursive decomp as
                    (n^k, ..., n^k, c) -> (n^l, ..., n^l, n^(k-l), ..., n^(k-l), c)
                    with numel = cn^(kd) (i.e. d copies of n^k, and then d copies of n^l and n^(k-l))
                    then we know every group is the subgroup of a symmetric
                    group. and we know the symmetric group has something to do
                    with GL_n. then lets just sample our differentials from I +
                    epsilon*N(0, 1) or something. Then we warp the mesh
                    according to the projective transformation (see homography
                    computation code, can directly use that here for the 2d
                    image case)

            sampling from a lie algebra, and letting the group element act on
            the coordinate. i guess then we probably want to do some map
            through rep.theory to go from group theory land back to tensors
            where we can do cross attention operations .. or something .. thing
            is we really would like some hierarchical representations


            TODO think about passing input into top 3 channels of the full
            (n^k,n^k) state, then having step(..., L) only propegate information
            through the last L+3:-1 channels .. this will allow for propegating
            with large L while not disrupting the low frequency image structure
            stored in earlier channel dims.

            observe that we can also introduct more interesting
            distributions for sampling from the lie algebra. In this way we can
            achieve a sort of weighted graph. then i think what would be ideal
            is to let some part of the state parameterize the distributions,
            with each scale L having a different distribution associated with
            it, and likely having the distribution for level L be
            parameteerized by a lower frequency part of the state. in this way
            we can achieve a sort of stability where low frequency parts of the
            state, which have more 'inertia' inform the flow through higher
            frequency parts of the state.

            TODO rewrite this whole fwd pass thing, go for simple impl, simple is good fast and correct.

            observe: probably dont need to antialias in downsample operations
            since randomized smoothing should come from randomly sampling the translations

        """

        N, K, C = self.N, self.K, self.C

        # pass to L-nested view
        state = self.state
        # if L > 0: # only operate on lower frequency parts .. something about this feels pretty hacky, TODO look at this some more
        #     state = state[..., 3+L:] # dont operate on the lower frequencies . this impl has just one hidden channel for each L, perhaps want to have more , TODO design decision
        state = state.reshape(N**L, N**(K-L), N**L, N**(K-L), C)
        state = state.permute(0, 2, 1, 3, 4) # (N^L, N^L, N^(K-L), N^(K-L), C)

        ## TODO instead of this random (dx, dy) shift, prefer to take some 3x3 matrix and warp a mesh over R^2 like in image warping i.e. warp via a homography.
        shifts = torch.randint(low=-1, high=2, size=(2,)).tolist()

        nei = state.roll(
            shifts=shifts, dims=(0, 1)
        )

        nei = nei.reshape(
            N**(2*L), N, N**(K-L-1), N, N**(K-L-1), C
        ).permute(0, 1, 3, 2, 4, 5
        ).reshape(N**(2*L), N**2, C*N**(2*(K-L-1)))

        state = state.reshape(
            N**(2*L), N, N**(K-L-1), N, N**(K-L-1), C
        ).permute(0, 1, 3, 2, 4, 5  # (N**(2*L), N, N, N**(K-L-1), N**(K-L-1), C)
        ).reshape(N**(2*L), N**2, C*N**(2*(K-L-1)))

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
        ) # (N**(2*L), N**2, C*N**(2*(K-L-1)))

        dxdt = dxdt.reshape(N**(2*L), N, N, N**(K-L-1), N**(K-L-1), C
        ).permute(0, 1, 3, 2, 4, 5 # (N**(2*L), N, N**(K-L-1), N, N**(K-L-1), C)
        ).reshape(N**L, N**L, N**(K-L), N**(K-L), C
        ).permute(0, 2, 1, 3, 4
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

        self.state += alpha * dxdt
        self.proj_to_torus()
