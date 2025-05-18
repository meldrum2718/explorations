import torch

from .coupled_attn_net import CAN
from ..utils import inspect


def test_batch_dim_independence(
    B = 10,
    C = 3,
    H = 32,
    W = 32,
    N = 9,
    n_steps = 10,
    test_batches = [2, 6, 2]
):
    can = CAN(
        B = B,
        C = C,
        H = H,
        W = W,
        N = N,
    )

    can2 = CAN(
        B = len(test_batches),
        C = C,
        H = H,
        W = W,
        N = N,
    )

    can2.state = torch.stack([can.state[:, b] for b in test_batches], dim=1)

    for t in range(n_steps):
        can.step()
        can2.step()
        batch_dim_indep = torch.all(
            torch.stack(
                [can.state[:, test_batches[b]] == can2.state[:, b] for b in range(len(test_batches))]
            )
        )
        assert batch_dim_indep, 'Non independence detected across batch dim'
    print('Passed batch dim independence test!')



def main():
    test_batch_dim_independence()



if __name__ == '__main__':
    main()
