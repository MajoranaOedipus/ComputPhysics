from .Interpolation import interpolate_Lagrange

"""Numerical integrations"""
from .Interpolation import interpolate_Lagrange
from .Polynomial import polynomial_integrate
from numbers import Number

def Newton_Cotes(f, a: Number, b: Number, n: int = 4):
    xs = [i/n for i in range(n+1)]
    ys = [f(a + x * (b - a)) for x in xs]
    Lagrange_poly = interpolate_Lagrange(xs, ys)
    return polynomial_integrate(Lagrange_poly)(1) * (b - a)