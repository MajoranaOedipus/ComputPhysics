"""Numerical integrations"""
from typing import Callable, Iterable, List, Union, Optional
from .Interpolation import interpolate_Lagrange
from .Polynomial import polynomial_integrate
from numbers import Integral, Number

SUPPORTED_METHOD = (
    "Romberg",
    "Newton-Cotes",
    "midpoint",
    "trapezoid",
    "Simpson"
    )

def quad(f: Callable[[Number], Number], a: Number, b: Number, method: str = "Romberg", *args, **kwargs) -> Number:
    """Numerical integration of f in [a, b]. This is just a user interface to specific methods.

    Args:
        f (Callable[[Number], Number]): the function to be integrate
        a (Number): the starting point of the integration
        b (Number): the end point of the integration
        method (str, optional): Integration methods, must be one of 
            ["Romberg", "Newton-Cotes", "trapezoid", "midpoint", "trapezoid", "Simpson"]. 
        Defaults to "Romberg".
        *args, **kwargs: args to be passed to the corresponding functions for methods

    Returns:
        Number: The integration result.
    """
    method = method.lower()
    if method == "romberg":
        return Romberg(f, a, b, *args, **kwargs)
    elif method in ["newton cotes", "newton-cotes", "newton_cotes"]:
        return Newton_Cotes(f, a, b, *args, **kwargs)
    elif method in ["midpoint", "mid-point", "mid point"]:
        return midpoint(f, a, b, *args, **kwargs)
    elif method == "trapezoid":
        return trapezoid(f, a, b, *kwargs, **kwargs)
    elif method == "simpson":
        return Simpson(f, a, b, *args, **kwargs)
    else:
        raise ValueError("Unkown method: {}. Must be one of the {}".format(method, SUPPORTED_METHOD))



def Newton_Cotes(f: Callable[[Number], Number], a: Number, b: Number, n: int = 2, 
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

def Newton_Cortes_equal_spaces(f: Callable[[Number], Number], a: Number, b: Number, n: int = 2, 
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

def midpoint(f: Callable[[Number], Number], a: Number, b: Number, N: int = 10) -> Number:
    """Use mid-point rule to calculate the numerical integration of function f in [a, b], with N-th order precision.

    Args:
        f (Callable[[Number], Number]): The function to be integrated
        a (Number): the starting point
        b (Number): the end point
        N (int, optional): the algebraic precision. Defaults to 10.

    Returns:
        Number: the integration result
    """
    dx = (b - a) / N
    ys = (f(a + dx/2 + dx * i) for i in range(1, N))
    return sum(ys) * dx

def trapezoid(f: Callable[[Number], Number], a: Number, b: Number, N: int = 10) -> Number:
    """Use trapezoid rule to calculate the numerical integration of function f in [a, b], with N-th order precision.

    Args:
        f (Callable[[Number], Number]): The function to be integrated
        a (Number): the starting point
        b (Number): the end point
        N (int, optional): the algebraic precision. Defaults to 10.

    Returns:
        Number: the integration result
    """
    dx = (b - a) / N  
    ys = (f(a + dx * i) for i in range(1, N))
    return dx/2 * (f(a) + f(b) + 2 * sum(ys))

def Simpson(f: Callable[[Number], Number], a: Number, b: Number, N: int = 10) -> Number:
    """Use Simpson rule to calculate the numerical integration of function f in [a, b], with N-th order precision.

    Args:
        f (Callable[[Number], Number]): The function to be integrated
        a (Number): the starting point
        b (Number): the end point
        N (int, optional): the algebraic precision. Defaults to 10.

    Returns:
        Number: the integration result
    """
    dx  = (b - a) / (2 * N)
    ys_odd = (f(a + dx * (2*i + 1)) for i in range(N))
    ys_even = (f(a + dx * 2*i) for i in range(1, N))
    return dx / 3 * (f(a) + f(b) + 4 * sum(ys_odd) + 2 * sum(ys_even))

def Romberg(f: Callable[[Number], Number], a: Number, b: Number, N: int = 10) -> Number:
    """Use Romberg method to calculate the numerical integration of function f in [a, b], with N-th order precision.

    Args:
        f (Callable[[Number], Number]): The function to be integrated
        a (Number): the starting point
        b (Number): the end point
        N (int, optional): the algebraic precision. Defaults to 10.

    Returns:
        Number: the integration result
    """
    h = [(b - a)/2**i for i in range(N)]
    R_col = [(f(a) + f(b))/2 * h[0]]
    for i in range(N - 1):
        R_new = R_col[i] / 2 + h[i+1] * sum(
            f(a + (2*k + 1) * h[i+1]) for k in range(2**i)
        )
        R_col.append(R_new)

    for j in range(1, N-1):
        R_col_new = []
        for i in range(1, N - j + 1):
            R_new = (4**j * R_col[i] - R_col[i-1])/(4**j - 1)
            R_col_new.append(R_new)
        R_col = R_col_new

    return R_col[0]


def Riemann_sum(f: Callable[[Number], Number], 
        partition: List[Number], 
        points: Optional[Iterable[Number]] = None) -> Number:
    """Calculate the Riemann sum of a function in given partition and at optional points (default to use left-point rule):
        Riemann_sum(f, P, x) = sum(f(x_i) * (P[i+1] - P[i])).

    Args:
        f (Callable[[Number], Number]): the function to be integrate
        partition (List[Number]): a collection of points in an inteval, with closed boundary 
            (i.e. 3 points is given if the interval is divided by 2)
        points (Optional[Iterable[Number]], optional): If None is given, then use the left-point. Defaults to None.

    Returns:
        Number: The Riemann sum
    """
    if points is None:
        points = partition[:-1]  # Use left-point Reimann rule
    return sum((f(x_i) * (partition[i+1] - partition[i]) for i, x_i in enumerate(points)))

def generate_partition(a: Number, b: Number, N: int) -> List[Number]:
    """Divide an interval [a, b] by N parts in equal spaces.

    Args:
        a (Number): The start points of the interval
        b (Number): the end points of the interval
        N (int): The number of parts in which the interval is divided

    Returns:
        List[Number]: a list of points [a, a + (b-a)/N, ..., b - (b-a)/N, b]
    """
    dx = (b - a) / N
    return [a + i * dx for i in range(N + 1)]