from .Polynomial import Polynomial

def interpolate_Lagrange(xs: list, ys: list) -> Polynomial:
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