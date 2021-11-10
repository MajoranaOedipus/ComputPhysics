"""
    1D Interpolation
    interpolate_lagrange and interpolate_Hermite available now
"""
from numbers import Number
from math import factorial
from itertools import islice
from .Polynomial import Polynomial, zero_poly

def interpolate_Lagrange(xs: "list[Number]", ys: "list[Number]") -> Polynomial:
    """return a Lagrange polynomial interpolated with Newton difference method.

    Args:
        xs (list[Num]): points
        ys (list[Num]): values

    Raises:
        ValueError: raised when amount of points in xs and ys are different

    Returns:
        Polynomial: interpolated
    """
    div_table = div_diff(xs, ys)
    n = len(xs)
    L = zero_poly()
    for j in range(n):
        Newton_term = Polynomial([div_table[j][0]])
        for i in range(j):
            Newton_term *= Polynomial([-xs[i], 1])
        L += Newton_term
    return L

def interpolate_Hermite(xs: "list[Number]", ys: "list[list[Number]]") -> Polynomial: 
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

def div_diff(xs: "list[Number]", ys: "list[Number]"):
    """
    Generate a divided difference table iteratively for x and y points, 
    where xs must be different.
    """
    n = len(xs)
    if n != len(ys):
        raise ValueError("xs and ys should be of same length.")

    div_diff = [ys]
    # div_diff[i][j] = f[z[j], ..., z[i+j]] for j in range(0, n*N-i)
    # where z[j] := xs[j]
    for i in range(1, n):
        last_depth = div_diff[-1]
        next_depth = []
        for j in range(n - i):
            diff = last_depth[j+1] - last_depth[j]
            diff /= xs[i+j] - xs[j]
            next_depth.append(diff)

        div_diff.append(next_depth)

    return div_diff