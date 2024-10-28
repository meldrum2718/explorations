import numpy as np


def ceinsum(X, Y, C, color, verbose=False) -> np.ndarray:
    """ Return einsum of X and some subtensor of Y, with all indices
        'read' from C.

        X, ker are ndarrays with n = X.shape[i] = X.shape[j] = ker.shape[k] for all valid i,j,k.
        C is a 2d array.

    """
    def p(*args):
        if verbose:
            print(*args)

    n = X.shape[0]

    if color: # design decision. just averaging color channels of C.
        C = np.mean(C, axis=-1)

    xnd = len(X.shape)
    ynd = len(Y.shape)

    p('x.s', X.shape, 'xnd', xnd)
    p('y.s', Y.shape, 'ynd', ynd)
    p('c.s', C.shape)

    ch = C.shape[0]
    n_samples = 5
    sh = ch // n_samples # sampling height

    p('sh', sh)

    px = np.argsort(np.sum(C[0*sh:1*sh, 0:xnd], axis=0)) # X perm
    po = np.argsort(np.sum(C[1*sh:2*sh, 0:xnd], axis=0)) # out perm
    py = np.argsort(np.sum(C[2*sh:3*sh, 0:ynd], axis=0)) # Y perm

    p('px.s', px.shape)
    p('po.s', po.shape)
    p('py.s', py.shape)

    knd = np.argmax(np.sum(C[2*sh:3*sh, 0:ynd-2])) + 2 # number of kernel dimensions
    nki = ynd - knd # number of kernel indices

    pk = np.argsort(np.sum(C[3*sh:4*sh, 0:knd], axis=0)) # kernel perm

    ki = tuple(np.argmax(C[4*sh:4*sh+n, 0:nki], axis=0)) # kernel indices
    
    p('knd', knd)
    p('nki', nki)
    p('ki', ki)
    p('pk.s', pk)

    ker = Y.transpose(*py)

    p('ker1.s', ker.shape)
    ker = ker[ki]
    p('ker.s', ker.shape)

    out = np.einsum(X, px, ker, pk, po)

    p('out.s', out.shape)
    p()

    return out


class RTN:
    def __init__(self, n, k, batch_dim=0, color=False, clip_min=-2, clip_max=2, verbose=False):
        """ initialize a random (n**k, n**k) state matrix.
        """

        def p(*args):
            if verbose:
                print(*args)

        self.n = n
        self.k = k
        self.batch_dim = batch_dim
        self.color = color
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.ndim = 2 * k

        self.flat_shape = (n**k, n**k)
        self.nested_shape = tuple([n for _ in range(2 * k)])

        self.flat_shape = (batch_dim,) + self.flat_shape
        self.nested_shape = (batch_dim,) + self.nested_shape

        if self.color:
            assert n >= 3, 'need n >=3 for color'
            self.flat_shape = self.flat_shape + (n,)
            self.nested_shape = self.nested_shape + (n,)
            self.ndim += 1

        p('flat shape', self.flat_shape)
        p('nested shape', self.nested_shape)

        self.X: np.ndarray = np.random.randn(*self.flat_shape).reshape(self.nested_shape)

    def output(self):
        """ read output from self.X

        ## TODO think about introducing conv stems for outpus. would encourage
        ## interpratible, spatially localized representations I think..  think this should be done outside the rtn..
                10/26/24. now, looking at this, it seems clear that conv stems
                are just a particular class of configurations of a network of
                matrices with ceinsum connections. TODO make this mapping
                explicit, take arbitrary network and express it as a ceinsum network.

                hmm, although outside the rtn might be wiser. in theory its
                very nice to think of the whole thing as an rtn with many
                subparts, but in practice it is good to make abstraction
                boundaries
        """
        if self.color:
            return self.X.reshape(self.flat_shape)[..., :3] # just use first three color channels for display
        return self.X.reshape(self.flat_shape)


    def step(self, ker, C, inp=None, alpha=0.01, noise_fbk=0, verbose=False) -> np.ndarray:
        """
        update: X += alpha * dXdt

        noise_fbk controls the size of the base noise, which is then scaled by
        prediction error and added to the state.

        """

        def p(*args):
            if verbose:
                print(*args)
        
        p('-----------')
        p('k.s', ker.shape)
        p('X.s', self.X.shape)
        p('C.s', C.shape)
        p('-----------')

        ker = ker.reshape(self.nested_shape)

        dX = np.empty_like(self.X)
        for b in range(self.batch_dim):
            dX[b] = ceinsum(self.X[b], ker[b], C[b], color=self.color, verbose=False)

        self.X = self.X + alpha * dX

        if noise_fbk > 0 and inp is not None:
            noise = np.random.normal(scale=noise_fbk, size=self.X.shape)
            err = np.linalg.norm(inp - self.output())
            noise = noise * err
            self.X = self.X + noise

        self.X = np.clip(self.X, self.clip_min, self.clip_max)

        return self.output()




    ## I dont think i want this..
    # def input(self, inp):
    #     """ inject inp into bottom right corner of self.X.
    #         TODO do this better, this feels crude.
    #     """
    #     if inp is None:
    #         print('trying to input None into rtn .. why tho?')
    #         return
    #     h, w = inp.shape[:2]
    #     x = self.X
    #     if self.batch_dim:
    #         x = x.reshape(self.n**self.batch_dim, self.n**k, self.n**self.batch_dim, self.n**k).transpose(0, 2, 1, 1)

    #     if self.color:
    #         x[-h:, -w:, :3] += inp
    #     else:
    #         x[-h:, -w:] += inp

    #     if self.batch_dim:
    #         # reshape back.
    #         x = x.reshape(self.n**self.batch_dim, self.n**k, self.n**self.batch_dim, self.n**k).transpose(0, 2, 1, 1)
    #         x = x.transpose(0, 2, 1, 3).reshape(self.X.shape)
    #         print('unbatching')

    #     self.X = x


    # def prod(self, ker, latent_shape, latent_perm) -> np.ndarray:
    #     """ reshape x0 into latent_shape, apply permutation, multiply by ker,
    #         then invert permutation and reshape back to original shape.
    #     """
    #     inv_perm = inv_p(latent_perm)
    #     x1 = self.X.reshape(*latent_shape).transpose(*latent_perm)
    #     out1 = x1 @ ker
    #     out0 = out1.transpose(*inv_perm).reshape(self.X.shape)
    #     return out0

