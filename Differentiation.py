from .LinearAlgebra import Matrix, concatenate, zeros, matrixify
from math import factorial

def central_diff_weights(Np: int, ndiv: int=1):
    if Np < ndiv + 1:
        raise ValueError("Number of points must be at least the derivative order + 1.")
    if Np % 2 == 0:
        raise ValueError("The number of points must be odd.")
    
    n_h = Np // 2 
    x = Matrix(list(range(-n_h, n_h + 1)), shape=Np) # A column vector: -n_h, -n_h + 1, ..., n_h
    X = zeros(Np) + 1.
    for k in range(1, Np):
        X = concatenate(X, matrixify(lambda x: x**k)(x))
    # Now each row of X is like [1, x_i, x_i**2, ..., x_i**(N_p - 1)], 
    # where x_i = -n_h + i
    weights = factorial(ndiv) * X.inverse()[ndiv]
    weights = weights.elements[0]
    return weights

def diff_f(f, dx: float = 1.0, n: int = 1, order: int = 3, *args, **kwargs):
    if order < n + 1:
        raise ValueError("'order' (the number of points used to compute the derivative), " + "must be at least the derivative order 'n' + 1.")
    if order % 2 == 0:
        raise ValueError("'order' (the number of points used to compute the derivative) " + "must be odd.")
    f_prime = None
    weights = central_diff_weights(order, n)

    def f_prime(x0: float):
        f_diff = 0.
        n_h = order // 2
        for i, w in enumerate(weights):
            f_diff += w * f(x0 + (i - n_h) * dx, *args, **kwargs)
        f_diff /= dx ** n
        return f_diff

    return f_prime

def diff(f, x0, dx=1.0, n=1, order=3, *args, **kwargs):
    f_diff = diff_f(f, dx, n, order, *args, **kwargs)(x0)
    return f_diff