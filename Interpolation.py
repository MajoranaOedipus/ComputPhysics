"""
    1D Interpolation
    interpolate_lagrange and interpolate_Hermite available now
"""
from typing import Union
from math import factorial
from itertools import islice
from .Polynomial import Polynomial

Num = Union[float, int]

def interpolate_Lagrange(xs: "list[Num]", ys: "list[Num]") -> Polynomial:
    """return a polynomial interpolated with Lagrange method.

    Args:
        xs (list[Num]): points
        ys (list[Num]): values

    Raises:
        ValueError: raised when amount of points in xs and ys are different

    Returns:
        Polynomial: interpolated
    """
    L = 0
    n = len(xs)
    if len(ys) != n:
        raise ValueError("The amount of points in xs and ys should be the same. ")
    for j in range(n):
        lj = 1
        for i in range(n):
            if i != j:
                lj *= Polynomial([- xs[i], 1]) / (xs[j] - xs[i])
        L += ys[j] * lj
    return L

def interpolate_Hermite(xs: "list[Num]", ys: "list[list[Num]]") -> Polynomial: 
    """
    Return the Hermite interpolation of xs and ys

    Args:
        xs (list): [x1, x2, ..., xN]
        ys (list): [[y1, y_diff1, ..., y_nth_diff1], ..., [yN, y_diffN, ..., y_nth_diffN]]

    Returns:
        Polynomial: The interpolated Hermite polynimial
    """
    N = len(xs)
    n = len(ys[0])

    # Generate a divided difference table iteratively
    div_diff = [[ys[j//n][0] for j in range(n*N)]]
    # div_diff[i][j] = f[z[j], ..., z[i+j]] for j in range(0, n*N-i)
    # where z[j] := xs[j//n]
    for i in range(1, n * N):
        last_depth = div_diff[-1]
        next_depth = []
        for j in range(n * N - i):
            if j//n == (i+j)//n:
                diff = ys[j//n][i]/factorial(i)
            else:
                diff = last_depth[j+1] - last_depth[j]
                diff /= xs[(i+j)//n] - xs[j//n]
            
            next_depth.append(diff)

        div_diff.append(next_depth)

    # We only need the first element of each depths:
    factors = [depth[0] for depth in div_diff]
    interpolated = Polynomial([factors[0]])
    term = Polynomial([1])
    for k, f in islice(enumerate(factors, -1), 1, None):
        term *= Polynomial([-xs[k//n], 1])
        interpolated += f * term   # prod(x - zj) for j in range(k)

    return interpolated