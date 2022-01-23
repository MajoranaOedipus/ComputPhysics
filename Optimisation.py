from numbers import Number
from typing import Callable, Optional, TypeVar

from .Differentiation import diff_f


X = TypeVar("X", bound=Number)
Y = TypeVar("Y", bound=Number)

def solve_binary(
    f: Callable[[X], Y], a: X = 0, b: X = 1, 
    TOL: float = 0, Nmax: int = 100, 
    *args) -> X:
    """Solve f(x) = 0 with binary search

    Args:
        f (Callable[[X], Y]): The function to be solved
        a (X, optional): The start point of the interval. Defaults to 0.
        b (X, optional): The end point of the interval. Defaults to 0.
        TOL (float, optional): Tolerent error of |f(x)|. Defaults to 0.
        Nmax (int, optional): Max step. Defaults to 100.

    Raises:
        ValueError: If a >= b
        Warning: If error is non-zero and Nmax reached, raised

    Returns:
        X: Numerical solution
    """
    for _ in range(Nmax):
        if a >= b:
            raise ValueError(f"b must be greater than a, but {a} >= {b}")
        x0 = (b + a) / 2
        y0 = f(x0, *args)
        y_left = f(a, *args)
        if abs(y0) <= TOL:
            break
        if y0 * y_left < 0:
            b = x0
        else:
            a = x0  
    else:
        if TOL:
            raise Warning(f"Max number of steps {Nmax} reached")

    return x0

def solve_iter(
    f: Callable[[X], Y], x: X = 0, 
    TOL: float = 0, Nmax: int = 100, 
    *args) -> X:
    """Solve f(x) = 0 with iteration (FPI)

    Args:
        f (Callable[[X], Y]): The function to be solved
        x (X, optional): Where to start iteration. Defaults to 0.
        TOL (float, optional): Tolerent error of |x|. Defaults to 0.
        Nmax (int, optional): Max step. Defaults to 100.

    Raises:
        Warning: If error is non-zero and Nmax reached, raised

    Returns:
        X: Numerical solution
    """
    def g(x):
        return f(x, *args) + x

    for _ in range(Nmax):
        x_next = g(x)
        if abs(x_next - x) <= TOL:
            break
        x = x_next
    else:
        if TOL:
            raise Warning(f"Max number of steps {Nmax} reached")
    return x_next

def solve_Newton(
    f: Callable[[X], Y], x: X = 0, 
    f_diff: Optional[Callable[[X], Y]] = None,
    p: int = 1,
    TOL: float = 0, Nmax: int = 10, 
    *args) -> X:
    """Solve f(x) = 0 with Newton's method.

    Args:
        f (Callable[[X], Y]): The function to be solved
        x (X, optional): Where to start iteration. Defaults to 0.
        f_diff (Callable[[X], Y], optional): the derivative function of f. 
            If None (default), then a numerical method is applied to calculate it (very costly).
            Note that `args` are not passed to f_diff if given.
        p (int, optional): used when the complexity of the root is p > 1.
        error (float, optional): Tolerent error of |dx|/|x|. Defaults to 0.
        Nmax (int, optional): Max step. Defaults to 10.
        *args: args to be passed to f 

    Raises:
        Warning: If error is non-zero and Nmax reached, raised

    Returns:
        X: Numerical solution
    """
    if f_diff is None:
        f_diff = diff_f(f, 1e-6, args=args)
    
    def g(x):
        return x - p * f(x, *args)/f_diff(x)

    for _ in range(Nmax):
        x_next = g(x)
        if abs((x_next - x) / x) <= TOL:
            break
        x = x_next
    else:
        if TOL:
            raise Warning(f"Max number of steps {Nmax} reached")
    return x_next

def solve_secant(
    f: Callable[[X], Y], x0: X = 0, x1: X = 1,
    p: int = 1,
    TOL: float = 0, Nmax: int = 10, 
    *args) -> X:
    """Solve f(x) = 0 with secant method.

    Args:
        f (Callable[[X], Y]): The function to be solved
        x0, x1 (X, optional): Where to start iteration. Defaults to 0 and 1.
        p (int, optional): used when the complexity of the root is p > 1.
        error (float, optional): Tolerent error of |dx|/|x|. Defaults to 0.
        Nmax (int, optional): Max step. Defaults to 10.
        *args: args to be passed to f 

    Raises:
        Warning: If error is non-zero and Nmax reached, raised

    Returns:
        X: Numerical solution
    """
    
    def g(x0, x1):
        return x1 - p * (x1 - x0) * f(x1, *args) / (f(x1, *args) - f(x0, *args))

    for _ in range(Nmax):
        try:
            x2 = g(x0, x1)
        except ZeroDivisionError:
            x2 = (x0 + x1)/2
        x0, x1 = x1, x2
        if abs((x1 - x0) / x0) < TOL:
                break
    else:
        if TOL:
            raise Warning(f"Max number of steps {Nmax} reached")
    return x2

def solve_regula_falsi(
    f: Callable[[X], Y], a: X = 0, b: X = 1,
    p: int = 1,
    TOL: float = 0, Nmax: int = 10, 
    *args) -> X:
    """Solve f(x) = 0 with regula falsi method.

    Args:
        f (Callable[[X], Y]): The function to be solved
        x0, x1 (X, optional): The start and end of the interval. Defaults to 0 and 1.
        TOL (float, optional): Tolerent error of |f(x)|. Defaults to 0.
        Nmax (int, optional): Max step. Defaults to 10.
        *args: args to be passed to f 

    Raises:
        Warning: If error is non-zero and Nmax reached, raised

    Returns:
        X: Numerical solution
    """
    f_origin = f
    f = lambda x: f_origin(x, *args)

    for _ in range(Nmax):
        c = (b * f(a) - a * f(b)) / (f(a) - f(b))
        if abs(f(c)) <= TOL:
            break
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    else:
        if TOL:
            raise Warning(f"Max number of steps {Nmax} reached")

    return c

# TODO: Muller, IQI and Brent