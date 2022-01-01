"""Numerical integrations"""
from typing import Callable, Union
from .Interpolation import interpolate_Lagrange
from .Polynomial import polynomial_integrate
from numbers import Integral, Number

def Newton_Cotes(f: Callable, a: Number, b: Number, n: int = 2, 
        spaces: Union[list, Integral] = 10) -> Number:
    """Using Newton-Cotes method to calculate the integration of a function f between a and b, with N equally spaced or given spaces.

    Args:
        f (Callable): the numerical function to be integrate 
        a (Number): the starting point of the integration
        b (Number): the end point of the integration
        n (int, optional): The number of samples points in every space. Defaults to 2.
        spaces (Union[list, Integral]): A collection of the divied points, if a integer is given, equally space the inteval. Defaults to 10.

    Returns:
        Number: The integration result
    """
    if isinstance(spaces, Integral):
        return Newton_Cortes_equal_spaces(f, a, b, n, spaces)
    
    ps = [i/n for i in range(n + 1)]    # divide 1 by n
    xs = spaces
    quad = 0
    for i in range(len(xs)):
        if i != 0:
            dx = xs[i] - xs[i - 1]
            ys = [f(xs[i - 1] + p * dx) for p in ps]
        else:
            dx = xs[i] - a
            ys = [f(a + p * dx) for p in ps]
        
        Lagrange_poly = interpolate_Lagrange(ps, ys)
        quad += polynomial_integrate(Lagrange_poly)(1) * dx
    return quad

def Newton_Cortes_equal_spaces(f: Callable, a: Number, b: Number, n: int = 2, 
        N: Integral = 10) -> Number:
    """Using Newton-Cotes method to calculate the integration of a function f between a and b, with equal spaces.

    Args:
        f (Callable): the numerical function to be integrate 
        a (Number): the starting point of the integration
        b (Number): the end point of the integration
        n (int, optional): The number of samples points in every space. Defaults to 2.
        N (Optional[list], optional): equally space the inteval into N parts. Defaults to 10.

    Returns:
        Number: The integration result
    """
    ps = [i/n for i in range(n + 1)]    # divide 1 by n
    dx = (b - a) / N
    xs = [dx * i for i in range(1, N + 1)]
    quad = 0
    for i in range(len(xs)):
        if i != 0:
            ys = [f(xs[i - 1] + p * dx) for p in ps]
        else:
            ys = [f(a + p * dx) for p in ps]
        
        Lagrange_poly = interpolate_Lagrange(ps, ys)
        quad += polynomial_integrate(Lagrange_poly)(1) * dx
    return quad