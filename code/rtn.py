import numpy as np

def inv_p(perm):
    """ from https://stackoverflow.com/questions/9185768/inverting-permutations-in-python """
    inv = np.empty_like(perm)
    inv[perm,] = np.arange(len(inv), dtype=inv.dtype)
    return inv   


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class RTN:
    """ Recurrent tensor network.
        X is an (n^k, n^k) array.
        We also think of X as having shape:
            (n,  n)^(k-1) (n x n)^(k-1)
            (n, n, n^2(k-1) )
            (n, n, (n, n, n^(2k-2) ) )

        and as an (n**2, n**2)**(k-1).

        Design decisions:
            how to perform the tensor contraction. reshaping or einsum. expect einsum more efficient, todo test
            how to do the sampling
            how to simulate the ode. a simple eulers method, or some ode solver from scipy/torchdiffeq.
            how to provide fbk. definitely some continuous signal, noise propto 'pred err'
            how to extract output. explicit or learned proj? cconv stems?
            how to connect to 'long term memory. almost certainly lonng term memory is  in the weights of an eg llm or some enc.dec.arch.'
            note updating wei of ltm should be a dyn.sys tht listens to the rtn. although also likely these trns are to be ran in parallel in some sparse, maybe 2 or 3 dim rgg?

        todo represent attn, ffn, cnn as specific instances or stable points of rtn.dyn.sys.

        believe they should have fairly clean / simpe representations.
        convolution with stride = kernel_size definitely is trivial to represent with the rtn.

    """

    def __init__(self, n, k):
        """ initialize a random n**k x n**k state matrix."""
        self.n = n
        self.k = k
        self.X = np.random.randn(n**k, n**k)

    def sample_prod_params(self):
        """ Use X to get arguments for a product of X with a subtensor of X.

            Just going to do a really simple sampling for now. future, perhaps
            make it be more built/structured, like with a conv stem or so

            Returns:
                (X_latent_perm, ker): a permutation to apply to X, and a kernel to multiply X by.
        """
        X, n, k = self.X, self.n, self.k

        nested_shape = (n for _ in range(2 * k))
        
        X_latent_perm = np.argsort(np.sum(X[0:n, 0:2*k], axis=0))

        ker_ndim = np.argmax(np.sum(X[n:2*n, 0:k-2], axis=0)) + 2
        ker_sample_perm = np.argsort(np.sum(X[2*n:3*n, 0:2*k], axis=0))
        ker_indices = tuple(np.argmax(X[3*n:4*n, 0:2*k-ker_ndim], axis=0))
        ker_latent_perm = np.argsort(np.sum(X[4*n:5*n, 0:ker_ndim], axis=0))
        
        ker = X.reshape(*nested_shape).transpose(*ker_sample_perm)[ker_indices].transpose(*ker_latent_perm)

        return X_latent_perm, ker


    def input(self, inp):
        """ inject inp into self.X. """
        h, w = inp.shape[:2]
        self.X[-h:, -w:] = inp

    def output(self):
        """ project state out to lower dimensional space.

        TODO think about introducing conv stems for outpus. would encourage
        interpratible, spatially localized representations I think..
        """
        pass

    def prod(self, ker, latent_shape, latent_perm) -> np.ndarray:
        """ reshape x0 into latent_shape, apply permutation, multiply by ker,
            then invert permutation and reshape back to original shape.
        """
        inv_perm = inv_p(latent_perm)
        x1 = self.X.reshape(*latent_shape).transpose(*latent_perm)
        out1 = x1 @ ker
        out0 = out1.transpose(*inv_perm).reshape(self.X.shape)
        return out0

    def step(self, inp=None, alpha=0.01) -> np.ndarray:
        """
        Euler step of dX/dt = X @ ker, where ker is a subtensor of X whos indices are determined by X.
        """
        if inp is not None:
            print('warning. inp being ignored (for now, not implemented yet)')

        latent_perm, ker = self.sample_prod_params()

        dX = self.prod(ker, (self.n for _ in range(2*self.k)), latent_perm=latent_perm)
        self.X = np.clip(self.X + alpha * dX, -2, 2)

        return self.X
