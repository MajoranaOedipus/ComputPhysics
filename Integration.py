
from ComputPhysics.Interpolation import interpolate_Lagrange

"""Numerical integrations"""
from .Interpolation import interpolate_Lagrange
from .Polynomial import polynomial_integrate
from numbers import Numbers

def Newton_Cotes(f, a: Numbers, b: Numbers, n: int = 4):
    xs = [i/n for i in range(n+1)]
    ys = [f(a + x * (b - a)) for x in xs]
    Lagrange_poly = interpolate_Lagrange(xs, ys)
    return polynomial_integrate(Lagrange_poly)(1) / (b - a)