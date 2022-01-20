
from numbers import Number
from typing import Any, Callable, Dict, List, Tuple, TypeVar, Union

T = TypeVar("T", bound=Number)
Y = TypeVar("Y", Number, List[Number])

Func = Callable[[T, Y, Any], Y]

def solve_IVP_explicit(
    f: Func[T, Y], y0: Y = 0, bounds: Tuple[T, T] = (0, 1),
    method: Union[str, Callable[[Func[T, Y], T, T, Y, Any], Y]] = "Euler", 
    N: int = 100, args: Tuple[Any] = ()) -> List[Y]:
    """Solve the IVP ODE problem y' = f(t, y) with initial condition y(t_0) = y_0
        using the given explicit method.

    TypeVars:
        T = TypeVar("T", bound=Number)
        Y = TypeVar("Y", Number, List[Number])
        Func = (T, Y, Any) -> Y
    Args:
        f (Func[T, Y], Y]): The function f in y' = f(t, y), 
            with first parametre one be the parametre of the system 
            and the second be a number or a list of number representing the function value.
        y0 (Y, optional): Initial value y(t_0). Defaults to 0.
        bounds (Tuple[T, T], optional): The range of parametres of the system. Defaults to (0, 1).
        N (int, optional): The number of steps in the given bounds. Defaults to 100.
        method (str | (Func[T, Y], T, T, Y, Any) -> Y], optional): The method. 
            It should be a str in SUPPORTED_METHODS, or a func (f, dt, t, y) -> next_y
        args (Tuple[Any], optional): Additional args to be passed to f. Defaults to ().

    Returns:
        List[Y]: A list of values of the solved function at t_i := t_0 + i * stepsize, for i in N
    """
    output = [y0]
    t_0, t_N = bounds
    dt: T = (t_N - t_0) / N

    if isinstance(method, str):
        method = method.lower()
        try:
            next_y = _SUPPORTED_METHODS[method]
        except KeyError:
            raise ValueError(f"method {method} not supported. must be one of the {SUPPORTED_METHODS}")

    for i in range(N-1):
        t_i: T = i * dt + t_0
        y_i = output[-1]
        y_ip1 = next_y(f, dt, t_i, y_i, *args)
        output.append(y_ip1)
    return output

def _next_y_Euler(f: Func[T, Y], dt: T, t: T, y: Y, *args)  -> Y:
    if isinstance(y, Number):
        next_y = y + dt * f(t, y, *args)
    else:
        next_y = [y_i + dt * f(t, y, *args)[i] for i, y_i in y]

    return next_y

def _next_y_trapezoid(f: Func[T, Y], dt: T, t: T, y: Y, *args)  -> Y:
    if isinstance(y, Number):
        y_predict = f(t, y, *args)
        y_next_predict = y + dt * y_predict
        next_y = y + dt/2 * (y_predict + f(t + dt, y_next_predict, *args))
    else:
        ys_predict = f(t, y, *args)
        ys_next_preict = [y_k + dt * ys_predict[k] for k, y_k in enumerate(y)]
        next_y = [y_i + dt/2 * (
                    ys_predict[i] + 
                    f(t + dt, ys_next_preict, *args)[i]) 
                for i, y_i in enumerate(y)]

    return next_y

def _next_y_midpoint(f: Func[T, Y], dt: T, t: T, y: Y, *args)  -> Y:
    if isinstance(y, Number):
        y_predict = f(t, y, *args)
        y_mid_predict = y + dt/2 * y_predict
        next_y = y + dt * f(t + dt/2, y_mid_predict, *args)
    else:
        ys_predict = f(t, y, *args)
        ys_mid_predict = [y_k + dt/2 * ys_predict[k] for k, y_k in enumerate(y)]
        next_y = [y_i + dt/2 * (
                    f(t + dt/2, ys_mid_predict, *args)[i]) 
                for i, y_i in enumerate(y)]

    return next_y

SUPPORTED_METHODS = [
    "Euler",
    "Midpoint",
    "Trapezoid"
]

_SUPPORTED_METHODS: Dict[str, Callable[[Func[T, Y], T, T, Y, Any], Y]] = {
    "euler": _next_y_Euler,
    "midpoint": _next_y_midpoint,
    "mid-point": _next_y_midpoint,
    "trapezoid": _next_y_trapezoid
}
