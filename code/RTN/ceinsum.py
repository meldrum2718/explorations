import numpy as np
import matplotlib.pyplot as plt

def ceinsum(X, Y, C) -> np.ndarray:
    """ Return einsum of X and some subtensor of Y, with all indices
        'read' from C.

        X, ker are ndarrays with n = X.shape[i] = X.shape[j] = ker.shape[k] for all valid i,j,k.
        C is a 2d array.

    """

    xnd = len(X.shape)
    ynd = len(Y.shape)

    print('x.s', X.shape, 'xnd', xnd)
    print('y.s', Y.shape, 'ynd', ynd)

    ch = C.shape[0]
    n_samples = 5
    sh = ch // n_samples # sampling height

    print('sh', sh)

    px = np.argsort(np.sum(C[0*sh:1*sh, 0:xnd], axis=0)) # X perm
    po = np.argsort(np.sum(C[1*sh:2*sh, 0:xnd], axis=0)) # out perm
    py = np.argsort(np.sum(C[2*sh:3*sh, 0:ynd], axis=0)) # Y perm

    print('px.s', px.shape)
    print('po.s', po.shape)
    print('py.s', py.shape)

    knd = np.argmax(np.sum(C[2*sh:3*sh, 0:ynd-2])) + 2 # number of kernel dimensions
    nki = ynd - knd # number of kernel indices
    ki = tuple(np.argmax(C[3*sh:4*sh, 0:nki], axis=0)) # kernel indices
    pk = np.argsort(np.sum(C[4*sh:5*sh, 0:knd], axis=0)) # kernel perm
    
    print('knd', knd)
    print('nki', nki)
    print('np.sum(C[3*sh:4*sh, 0:nki], axis=0).shape', np.sum(C[3*sh:4*sh, 0:nki], axis=0).shape)
    print('ki', ki)
    print('pk.s', pk)

    ker = Y.transpose(*py)[ki]
    print('ker.s', ker.shape)

    out = np.einsum(X, px, ker, pk, po)
    return out


n = 2
k = 3
X = np.random.randn(n**(2*k)).reshape(*[n for _ in range(2*k)])
Y = np.random.randn(n**(2*k)).reshape(*[n for _ in range(2*k)])
out = ceinsum(X, Y, X.reshape(n**k, n**k))

print(X.shape)
print(Y.shape)
print(out.shape)


fig, axs = plt.subplots(1, 3)
axs = axs.reshape(-1)
for ax in axs: ax.axis('off')
axs[0].imshow(X.reshape(n**k, n**k)); axs[0].set_title('X')
axs[1].imshow(Y.reshape(n**k, n**k)); axs[1].set_title('Y')
axs[2].imshow(out.reshape(n**k, n**k)); axs[2].set_title('out')

plt.show()
