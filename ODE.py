
from typing import Any, Callable, List, Tuple, TypeVar, Union
from numbers import Number
from .LinearAlgebra import Matrix

T = TypeVar("T", bound=Number)
Y = TypeVar("Y", Number, List[Number])

def solve_IVP_Euler_explicit(
    f: Callable[[T, Y, Any], Y], y0: Y = 0, bounds: Tuple[T, T] = (0, 1), 
    N: int = 100, args: Tuple[Any] = ()) -> List[Y]:
    """Solve the IVP ODE problem y' = f(t, y) with initial condition y(t_0) = y_0
        using explicit Euler method.

    TypeVars:
        T = TypeVar("T", bound=Number)
        Y = TypeVar("Y", Number, List[Number])
    Args:
        f (Callable[[T, Y, Any], Y]): The function f in y' = f(t, y), 
            with first parametre one be the parametre of the system 
            and the second be a number or a list of number representing the function value.
        y0 (Y, optional): Initial value y(t_0). Defaults to 0.
        bounds (Tuple[T, T], optional): The range of parametres of the system. Defaults to (0, 1).
        N (int, optional): The number of steps in the given bounds. Defaults to 100.
        args (Tuple[Any], optional): Additional args to be passed to f. Defaults to ().

    Returns:
        List[Y]: A list of values of the solved function at t_i := t_0 + i * stepsize, for i in N
    """
    output = [y0]
    t_0, t_N = bounds
    dt: T = (t_N - t_0) / N

    if isinstance(y0, Number):
        def next_y(t_i: Union[Number, Matrix], y_i: Y, args: Any) -> Y:
            return y_i + dt * f(t_i, y_i, *args)
    else:
        def next_y(t_i: Union[Number, Matrix], y_i: Y, args: Any) -> Y:
            return [y_ij + dt * f(t_i, y_i, *args)[j] for j, y_ij in enumerate(y_i)]

    for i in range(N-1):
        t_i: T = i * dt + t_0
        y_i = output[-1]
        y_ip1 = next_y(t_i, y_i, args)
        output.append(y_ip1)
    return output

def solve_IVP_trapezoid_explicit(
    f: Callable[[T, Y, Any], Y], y0: Y = 0, bounds: Tuple[T, T] = (0, 1), 
    N: int = 100, args: Tuple[Any] = ()) -> List[Y]:
    """Solve the IVP ODE problem y' = f(t, y) with initial condition y(t_0) = y_0
        using explicit trapezoid method.

    TypeVars:
        T = TypeVar("T", bound=Number)
        Y = TypeVar("Y", Number, List[Number])
    Args:
        f (Callable[[T, Y, Any], Y]): The function f in y' = f(t, y), 
            with first parametre one be the parametre of the system 
            and the second be a number or a list of number representing the function value.
        y0 (Y, optional): Initial value y(t_0). Defaults to 0.
        bounds (Tuple[T, T], optional): The range of parametres of the system. Defaults to (0, 1).
        N (int, optional): The number of steps in the given bounds. Defaults to 100.
        args (Tuple[Any], optional): Additional args to be passed to f. Defaults to ().

    Returns:
        List[Y]: A list of values of the solved function at t_i := t_0 + i * stepsize, for i in N
    """
    output = [y0]
    t_0, t_N = bounds
    dt: T = (t_N - t_0) / N

    if isinstance(y0, Number):
        def next_y(t_i: Union[Number, Matrix], y_i: Y, args: Any) -> Y:
            y_predict = f(t_i, y_i, *args)
            return y_i + dt/2 * (y_predict + f(t_i + dt, y_i + dt * y_predict, *args))
    else:
        def next_y(t_i: Union[Number, Matrix], y_i: Y, args: Any) -> Y:
            ys_predict = f(t_i, y_i, *args)
            return [y_ij + dt/2 * (
                    ys_predict[j] + 
                    f(t_i + dt, [y_ik + dt * ys_predict[k] for k, y_ik in enumerate(y_i)], *args)[j]) 
                for j, y_ij in enumerate(y_i)]

    for i in range(N-1):
        t_i: T = i * dt + t_0
        y_i = output[-1]
        y_ip1 = next_y(t_i, y_i, args)
        output.append(y_ip1)
    return output