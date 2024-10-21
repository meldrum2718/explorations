import numpy as np

def inv_p(perm):
    """ from https://stackoverflow.com/questions/9185768/inverting-permutations-in-python """
    inv = np.empty_like(perm)
    inv[perm,] = np.arange(len(inv), dtype=inv.dtype)
    return inv   


class RTN:
    """ Recurrent tensor network.
        X is an (n^k, n^k) array.
        We also think of X as having shape:
            (n,  n)^(k-1) (n x n)^(k-1)
            (n, n, n^2(k-1) )
            (n, n, (n, n, n^(2k-2) ) )

        and as an (n**2, n**2)**(k-1).
    """

    def __init__(self, n, k, color=False, clip_min=-2, clip_max=2):
        """ initialize a random (n**k, n**k) state matrix."""
        self.n = n
        self.k = k
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.color = color
        if self.color: # add a color dim
            self.X = np.random.randn(n**k, n**k, n)
        else:
            self.X = np.random.randn(n**k, n**k)

    def input(self, inp):
        """ inject inp into bottom right corner of self.X.
            TODO do this better, this feels crude.
        """
        if inp is None:
            return
        h, w = inp.shape[:2]
        self.X[-h:, -w:] = inp

    def output(self, h, w):
        """ project state out to lower dimensional space.
        for now just done by returning bottom right (h, w) submatrix.

        TODO think about introducing conv stems for outpus. would encourage
        interpratible, spatially localized representations I think..

        But for now, we just pull output from some submatrix of the state, and
        let the rtn implicitly find good representations. if we have a regular
        sampling period, this should work out nicely.
        """
        return self.X[-h:, -w:]

    def prod(self, ker, latent_shape, latent_perm) -> np.ndarray:
        """ reshape x0 into latent_shape, apply permutation, multiply by ker,
            then invert permutation and reshape back to original shape.
        """
        inv_perm = inv_p(latent_perm)
        x1 = self.X.reshape(*latent_shape).transpose(*latent_perm)
        out1 = x1 @ ker
        out0 = out1.transpose(*inv_perm).reshape(self.X.shape)
        return out0

    def step(self, latent_perm, ker, inp=None, alpha=0.01, sigma=0) -> np.ndarray:
        """
        Let dXdt = X @ ker, then update: X += alpha * dXdt

        sigma is a 'noise parameter', paremeterizing the stddev of the base
        noise, which is scaled by prediction error and added to the model.

        """
        if inp is not None:
            self.input(inp)

        if self.color:
            latent_shape = (self.n for _ in range(2*self.k + 1))
        else:
            latent_shape = (self.n for _ in range(2*self.k))
        dX = self.prod(ker, latent_shape=latent_shape, latent_perm=latent_perm)
        self.X = self.X + alpha * dX
        # self.input(inp)

        if (inp is not None) and (sigma > 0):
            noise = np.random.normal(scale=sigma, size=self.X.shape)
            err = np.linalg.norm(inp - self.output(*inp.shape[:2]))
            noise = noise * err
            self.X = self.X + noise

        self.X = np.clip(self.X, self.clip_min, self.clip_max)

        return self.X
