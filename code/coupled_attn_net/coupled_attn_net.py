import torch
import torch.nn.functional as F
import networkx as nx


from ..utils import inspect, normalize

class CAN:
    """
    A graph where the nodes are images, and the edges are attention
    connections. Updates through time as a diffeq.

    Design decisions:
        how to parameterize the network topology? do predefined? or let some
            part of the state determing the network topology. almost certainly
            the second. i suppose included in the network topology is not only
            the edges, but the edge weights, which key, query, and value they
            use
        how to parameterize the step size. presumably want the step size to be
        different in different parts of the network. there is some chance that
        simply a difference in ordero of magnitude of the states will provide
        this difference in inertia that we are looking for, but i kinda feel
        like it could be better to explicitly give the model the ability to condition their ideas 

    """
    def __init__(
        self,
        B: int,
        C: int,
        H: int,
        W: int,
        G: nx.DiGraph,
        clip_min: int = -2,
        clip_max: int = 2,
        verbose: bool = False
    ):

        self.verbose = verbose

        def p(*args):
            if self.verbose:
                print(*args)

        self.B = B
        self.C = C
        self.H = H
        self.W = W
        self.G = G
        self.n_nodes = G.number_of_nodes()

        self.clip_min = clip_min
        self.clip_max = clip_max

        self.state = torch.randn(self.n_nodes, B, C, H, W)

        # ## TODO do this bit better. at least dont have it fixed and inflexible. think about doing different sized states for wei, ker, c_idx. perhaps an interpolation (graphon like)
        # self.wei_idx = 0
        # self.ker_idx = 1
        # self.C_idx = 2
        # self.stdout_idx = 3
        # self.stdin_idx = 4

    def output(self, node_idx):
        """ output from self.state[node_idx], ready for matplotlib.
        """
        if self.C >= 3:
            out = self.state[node_idx, :, 0:3, :, :] # just use first three color channels for display for now
        else:
            out = torch.mean(self.state[node_idx], dim=1).unsqueeze(1) # mean across channel dim
        # out.shape = (B, C_out, H, W)
        out = out.detach().cpu().permute(0, 2, 3, 1) # (B, C_out, H, W) -> (B, H, W, C_out)
        return normalize(out)

    def input(self, inp, node_idx, alpha=1):
        """ update state[idx] as a convex combination with inp."""
        H, W, C = inp.shape

        inp = inp.permute(2, 0, 1).unsqueeze(0) # (H, W, C) -> (1, C, H, W)

        if self.C >= C: # if more channels in self.state than in inp
            self.state[node_idx][:, 0:C, :, :] = alpha * inp  +  (1 - alpha) * self.state[node_idx][:, 0:C, :, :]
        elif self.C == 1:
            inp = torch.mean(inp, dim=1) # (1, C, H, W) -> (1, 1, H, W)
            self.state[node_idx] = alpha * inp  +  (1 - alpha) * self.state[node_idx]
        else:
            raise Exception('TODO have not handled case where self.state has 1 channel and inp has more channels')


    def step(self, alpha=0.01):
        """
        update: X += alpha * dXdt

        ## TODO make alpha be something more interesting than just slider
        controlled probably.. lets let the network parameterize this i think.
        or potentially, when building more hand cratfted architectures, would
        presumably like to be able to control the 'inertia' (alpha) of the
        various nodes.. 
        """

        def p(*args):
            if self.verbose:
                print(*args)

        # just loop over for now.. wonder if we save much by vectorizing..
        # probably when n_nodes grows a bit.. definitely in fact.. ah well
        # going to just implement first in python.

        ## TODO make this better. really basically unusuable for higher batch
        ## dim and higher number of nodes. this impl does not scale.
        

        for u, v in self.G.edges:
            self.state[v] += alpha * F.scaled_dot_product_attention(
                self.state[u].permute(0, 1, 3, 2),
                self.state[u].permute(0, 1, 3, 2),
                self.state[v].permute(0, 1, 3, 2)
            )

            self.state = torch.clip(self.state, self.clip_min, self.clip_max)













    ## def wei(self, i, j):
    ##     """ Return the weight of the edge (i -> j)

    ##         have 0 <= i, j <= n_nodes

    ##         assume n_nodes <= n**k

    ##         then can very simply just read from the top left
    ##         [:n_nodes, :n_nodes] submatrix.
    ##     """
    ##     W = self.nodes[self.wei_idx]
    ##     return torch.mean(W[:, i, j]) # average over channel dim


    ## def ker(self, b, i, j):
    ##     """ Return the kernel associated with the edge (i -> j)
    ##         Hence we need an index 0 <= idx <= n_nodes.

    ##         assume n_nodes**2 <= n**k

    ##         we are looking for the adjacency matrix of a matching in a
    ##         bipartite graph between edges and nodes then just need (n_nodes**2,
    ##         n_nodes) 0-1 matrix with all rows summing to 1.

    ##         then:
    ##     """
    ##     K = self.nodes[self.ker_idx]
    ##     K = K[b] # index into batch dim
    ##     K = torch.mean(K, dim=-1) # average over channel dim
    ##     K = K[0:self.n_nodes**2, 0:self.n_nodes]
    ##     K = torch.argmax(K, dim=-1)
    ##     K = K.reshape(self.n_nodes, self.n_nodes)
    ##     return self.nodes[K[i, j]][b]


    ## def C(self, i, j):
    ##     """ same procedure as for above """
    ##     _C = self.nodes[self.C_idx]
    ##     _C = _C[b] # index into batch dim
    ##     _C = torch.mean(_C, dim=-1) # average over channel dim
    ##     _C = torch.argmax(_C[0:self.n_nodes**2, 0:self.n_nodes], dim=-1).reshape(self.n_nodes, self.n_nodes)
    ##     return self.nodes[_C[i, j]]



# def ceinsum(X, Y, C, color, verbose=False) -> np.ndarray:
#     """ Return einsum of X and some subtensor of Y, with all indices
#         'read' from C.
# 
#         X, ker are ndarrays with n = X.shape[i] = X.shape[j] = ker.shape[k] for all valid i,j,k.
#         C is a 2d array.
# 
#     """
#     def p(*args):
#         if verbose:
#             print(*args)
# 
#     n = X.shape[0]
# 
#     if color: # design decision. just averaging color channels of C.
#         C = np.mean(C, axis=-1)
# 
#     xnd = len(X.shape)
#     ynd = len(Y.shape)
# 
#     p('x.s', X.shape, 'xnd', xnd)
#     p('y.s', Y.shape, 'ynd', ynd)
#     p('c.s', C.shape)
# 
#     ch = C.shape[0]
#     n_samples = 5
#     sh = ch // n_samples # sampling height
# 
#     p('sh', sh)
# 
#     px = np.argsort(np.sum(C[0*sh:1*sh, 0:xnd], axis=0)) # X perm
#     po = np.argsort(np.sum(C[1*sh:2*sh, 0:xnd], axis=0)) # out perm
#     py = np.argsort(np.sum(C[2*sh:3*sh, 0:ynd], axis=0)) # Y perm
# 
#     p('px.s', px.shape)
#     p('po.s', po.shape)
#     p('py.s', py.shape)
# 
#     knd = np.argmax(np.sum(C[2*sh:3*sh, 0:ynd-2])) + 2 # number of kernel dimensions
#     nki = ynd - knd # number of kernel indices
# 
#     pk = np.argsort(np.sum(C[3*sh:4*sh, 0:knd], axis=0)) # kernel perm
# 
#     ki = tuple(np.argmax(C[4*sh:4*sh+n, 0:nki], axis=0)) # kernel indices
#     
#     p('knd', knd)
#     p('nki', nki)
#     p('ki', ki)
#     p('pk.s', pk)
# 
#     ker = Y.transpose(*py)
# 
#     p('ker1.s', ker.shape)
#     ker = ker[ki]
#     p('ker.s', ker.shape)
# 
#     out = np.einsum(X, px, ker, pk, po)
# 
#     p('out.s', out.shape)
#     p()
# 
#     return out

