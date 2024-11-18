import torch

from ..utils import inspect

class RTN:
    """
    A graph like object, G = (V, E). Simulates a dynamical system, where for v in V,
        dv/dt = sum_{e = (u, v)} w(e) * ceinsum(u, e_ker, e_C)

    Again, in the theme of lettting the dyn.sys control most parameters of the
    dyn.sys, lets just parameterize w(e), e_ker, e_C with nodes in the network.

    Observe that this structure allows for:
        - kernel and control matrix sharing across the network
        - the desirable property of having feedback flows (e.g. deeper representations can influence the processing of shallow representations)
    """
    def __init__(self, n, k, n_nodes, batch_dim=1, color=False, clip_min=-2, clip_max=2, verbose=False):
        """ initialize a random (n**k, n**k) state matrix.
        """
        self.verbose = verbose

        def p(*args):
            if self.verbose:
                print(*args)

        self.n = n
        self.k = k
        self.batch_dim = batch_dim
        self.color = color
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.flat_shape = (n**k, n**k)
        self.nested_shape = (n,) * (2*k)

        if self.color:
            assert n >= 3, 'need n >=3 for color'
            self.flat_shape = self.flat_shape + (n,)
            self.nested_shape = self.nested_shape + (n,)
        else:
            # lets always have a channel dim at the end
            self.flat_shape = self.flat_shape + (1,)
            self.nested_shape = self.nested_shape + (1,)

        self.ndim = len(self.nested_shape)

        p('flat shape', self.flat_shape)
        p('nested shape', self.nested_shape)

        assert 5 <= n_nodes**2 <= n**k

        self.n_nodes = n_nodes
        self.nodes = torch.randn(n_nodes, batch_dim, *self.flat_shape) # 

        ## TODO do this bit better. at least dont have it fixed and inflexible. think about doing different sized states for wei, ker, c_idx. perhaps an interpolation (graphon like)
        self.wei_idx = 0
        self.ker_idx = 1
        self.C_idx = 2
        self.stdout_idx = 3
        self.stdin_idx = 4

    def output(self, idx=None):
        """ read output from self.X

        ## TODO think about introducing conv stems for outputs. would encourage
        ## interpratible, spatially localized representations I think..  think this should be done outside the rtn..
                10/26/24. now, looking at this, it seems clear that conv stems
                are just a particular class of configurations of a network of
                matrices with ceinsum connections. TODO make this mapping
                explicit, take arbitrary network and express it as a ceinsum network.

                hmm, although outside the rtn might be wiser. in theory its
                very nice to think of the whole thing as an rtn with many
                subparts, but in practice it is good to make abstraction
                boundaries

                but also feel like rtn should be able to find the appropriate conv stems if given enough expressivity / direction
                indeed reimplementing rtn as a network of ceinsum connections will hopefully take care of this.
        """
        if idx is None:
            idx = self.stdout_idx
        if self.color:
            return self.nodes[idx][..., :3] # just use first three color channels for display for now
        return self.nodes[idx]

    def input(self, inp, idx=None, alpha=1):
        """ update nodes[idx] as a convex combination with inp. """
        H, W, C = inp.shape
        if self.color:
            assert C == 3
        else:
            assert C == 1

        if idx is None:
            idx = self.stdin_idx

        inp = inp.unsqueeze(0) # put a batch dim in there

        if self.color:
            self.nodes[idx][..., 0:3] = alpha * inp[..., 0:3]  +  (1 - alpha) * self.nodes[idx][..., 0:3]
        else:
            self.nodes[idx] = alpha * inp  +  (1 - alpha) * self.nodes[idx]

    def ceinsum(self, X, Y, C) -> torch.Tensor:
        """ Return einsum of X and some subtensor of Y, with all indices
            'read' from C.

            X, ker are ndarrays with n = X.shape[i] = X.shape[j] = ker.shape[k] for all valid i,j,k.
            C is a 2d array.
        """

        def p(*args):
            if self.verbose:
                print(*args)

        X = X.reshape(self.nested_shape)
        Y = Y.reshape(self.nested_shape)

        # looks like im already avering in the self.C() function whos output gets passed here
        ## if self.color: # TODO design decision. just averaging color channels of C for now.
        ##     ## TODO design decision. here, choosing to take a single control matrix C for all the batch dim. i.e. just averaging 
        C = torch.mean(C, dim=-1) # average over channel dim of C
        ##     C = torch.mean(C, dim=(0, 1)) # average over both batch and channel dim of C
        assert len(C.shape) == 2, 'expect C to be a 2d tensor for reading off einsum params'

        p('x.s', X.shape)
        p('y.s', Y.shape)
        p('c.s', C.shape)

        ch = C.shape[-2]
        n_samples = 5
        sh = ch // n_samples # sampling height

        p('sh', sh)

        ## here is where we could really benefit from downsampling C to be the right shape.. feels incredibly arbitrary to just take the first few columns of C as giving the parameters. why not using all of C?
        px = torch.argsort(torch.sum(C[0*sh:1*sh, 0:self.ndim], dim=0)) # X perm
        po = torch.argsort(torch.sum(C[1*sh:2*sh, 0:self.ndim], dim=0)) # out perm
        py = torch.argsort(torch.sum(C[2*sh:3*sh, 0:self.ndim], dim=0)) # Y perm

        p('px.s', px.shape)
        p('po.s', po.shape)
        p('py.s', py.shape)

        knd = torch.argmax(torch.sum(C[2*sh:3*sh, 0:self.ndim-2])) + 2 # number of kernel dimensions
        nki = self.ndim - knd # number of kernel indices

        pk = torch.argsort(torch.sum(C[3*sh:4*sh, 0:knd], dim=0)) # kernel perm

        ki = tuple(torch.argmax(C[4*sh:4*sh+self.n, 0:nki], dim=0)) # kernel indices
        
        p('knd', knd)
        p('nki', nki)
        p('ki', ki)
        p('pk.s', pk)

        ker = Y.permute(*py)

        p('ker1.s', ker.shape)
        ker = ker[ki]
        p('ker.s', ker.shape)

        out = torch.einsum(X, px, ker, pk, po)

        out = out.reshape(self.flat_shape)

        p('out.s', out.shape)
        p()

        return out

    def step(self, alpha=0.01):
        """
        update: X += alpha * dXdt

        ## TODO make alpha be something more interesting than just slider controlled probably.. lets let the network parameterize this i think. or potentially, when building more hand cratfted architectures, would presumably like to be able to control the 'inertia' (alpha) of the various nodes.. 
        """

        def p(*args):
            if self.verbose:
                print(*args)

        # just loop over for now.. wonder if we save much by vectorizing..
        # probably when n_nodes grows a bit.. definitely in fact.. ah well
        # going to just implement first in python.

        for i, x in enumerate(self.nodes):
            dxdt = torch.zeros_like(x)
            for j, nei in enumerate(self.nodes):
                for b in range(self.batch_dim):
                    w = self.wei(b, j, i)
                    #if np.abs(w) > 0.01: # TODO magic number
                    p('w', w)
                    p('nei[b].shape', nei[b].shape)
                    ker = self.ker(b, j, i)
                    C = self.C(b, j, i)
                    p('ker.shape', ker.shape)
                    p('C.shape', C.shape)
                    p('dxdt.shape', dxdt.shape)

                    dxdt[b] += w * self.ceinsum(nei[b], ker, C)
            x = x + alpha * dxdt
            x = torch.clip(x, self.clip_min, self.clip_max)
            self.nodes[i] = x

    def wei(self, b, i, j):
        """ Return the weight of the edge (i -> j)

            have 0 <= i, j <= n_nodes

            assume n_nodes <= n**k

            then can very simply just read from the top left
            [:n_nodes, :n_nodes] submatrix.
        """
        W = self.nodes[self.wei_idx]
        return torch.mean(W[b, i, j]) # average over channel dim


    def ker(self, b, i, j):
        """ Return the kernel associated with the edge (i -> j)
            Hence we need an index 0 <= idx <= n_nodes.

            assume n_nodes**2 <= n**k

            we are looking for the adjacency matrix of a matching in a
            bipartite graph between edges and nodes then just need (n_nodes**2,
            n_nodes) 0-1 matrix with all rows summing to 1.

            then:
        """
        K = self.nodes[self.ker_idx]
        K = K[b] # index into batch dim
        K = torch.mean(K, dim=-1) # average over channel dim
        K = K[0:self.n_nodes**2, 0:self.n_nodes]
        K = torch.argmax(K, dim=-1)
        K = K.reshape(self.n_nodes, self.n_nodes)
        return self.nodes[K[i, j]][b]


    def C(self, b, i, j):
        """ same procedure as for above """
        _C = self.nodes[self.ker_idx] ## TODO this should be self.C_idx .. going to leav it unchanged now because ideally want to recover identical behavior to the numpy implementation just in torch. But yeah fix this.
        _C = _C[b] # index into batch dim
        _C = torch.mean(_C, dim=-1) # average over channel dim
        _C = torch.argmax(_C[0:self.n_nodes**2, 0:self.n_nodes], axis=-1).reshape(self.n_nodes, self.n_nodes)
        return self.nodes[_C[i, j]][b]





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

