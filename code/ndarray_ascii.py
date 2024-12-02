import argparse
import numpy as np

"""
Ok, can get row major order in the ascii print if:
    pass shape = (n_1, n_2, ..., n_d)
        if d is even:
            pass --transpose (0, _ 1, _, 2, ... (d/2)-1, _), then fill the underscores with (d/2), (d/2)+1, (d/2)+2 ...
        if d is odd:
            pass --transpose (_ 0, _ 1, _, 2, ... (d/2)-1, _), then fill the underscores with (d/2), (d/2)+1, (d/2)+2 ...

        Note the above description may well have off by one errors, but the
        pattern is there. d is odd ==> put underscore before 0. Always have
        underscore at end.
"""


def get_2d_ascii(x: np.ndarray, max_width=None):
    """Generate and print a 2D ASCII table from x."""
    assert len(x.shape) == 2

    # Calculate the width for alignment based on the maximum number length
    if max_width is None:
        max_width = len(str(np.max(x))) + 1  # Extra space for better alignment

    s = '|'
    for row_idx, row in enumerate(x):
        ds = ''
        for value in row:
            ds += f'{str(value).ljust(max_width)}'
        s += ds + '|'
        if row_idx < x.shape[0] - 1:
            s += '\n|'
    hline = '+' + '-' * (s.index('\n') - 2) + '+'
    return hline  + '\n' + s + '\n' + hline


def get_nd_ascii(x: np.ndarray):
    def inner(x: np.ndarray, max_width, is_horiz):
        if len(x.shape) == 2:
            return get_2d_ascii(x, max_width)
        submatrices = [inner(x[i, ...], max_width, not is_horiz) for i in range(x.shape[0])]
        if is_horiz:
            nrows = len(submatrices[0].split('\n'))
            rows = ['|' for _ in range(nrows)]
            for submatrix in submatrices:
                for idx, line in enumerate(submatrix.split('\n')):
                    rows[idx] += line
            for idx in range(nrows):
                rows[idx] += '|'
            return '\n'.join(rows)
        else:
            return '\n'.join(submatrices)



    assert len(x.shape) > 1

    return inner(x, max_width=len(str(np.max(x))) + 1, is_horiz=len(x.shape) % 2 == 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig_shape', type=int, default=None, help='Original shape.', nargs='*')
    parser.add_argument('--transpose', type=int, nargs='*', help='Transposition to apply to reshaped ndarray.')
    args = parser.parse_args()

    arr = np.arange(np.prod(args.orig_shape)).reshape(args.orig_shape)
    if args.transpose:
        arr = arr.transpose(args.transpose)

    print(get_nd_ascii(arr))
