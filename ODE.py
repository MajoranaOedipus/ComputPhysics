from numbers import Number
from typing import Any, Callable, Dict, List, Tuple, TypeVar, Union, Literal

T = TypeVar("T", bound=Number)
Y = TypeVar("Y", Number, List[Number])

Func = Callable[[T, Y, Any], Y]

# def solve_IVP(
#     f: Func[T, Y], y0: Y = 0, bounds: Tuple[T, T] = (0, 1),
#     method: Union[str, Callable[[Func[T, Y], T, T, Y, Any], Y]] = "Euler", 
# ):
#     return 

def solve_IVP_explicit(
    f: Func[T, Y], y0: Y = 0, bounds: Tuple[T, T] = (0, 1),
    method: Union[str, Callable[[Func[T, Y], T, T, Y, Any], Y]] = "Euler", 
    N: int = 100, args: Tuple[Any] = ()) -> List[Y]:
    """Solve the IVP ODE problem y' = f(t, y) with initial condition y(t_0) = y_0
        using the given explicit method， with constant step (b - a)/N.

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
            It should be a str in SUPPORTED_METHODS, or a func (f, dt, t, y, *args) -> next_y
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
            next_y = _SUPPORTED_IVP_CONST_STEP_METHODS[method]
        except KeyError:
            raise ValueError(f"method {method} not supported. must be one of the {SUPPORTED_METHODS}")
    else:
        next_y = method

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

def _next_y_RK4(f: Func[T, Y], dt: T, t: T, y: Y, *args)  -> Y:
    if isinstance(y, Number):
        s1 = f(t, y, *args)
        s2 = f(t + dt/2, y + dt/2 * s1, *args)
        s3 = f(t + dt/2, y + dt/2 * s2, *args)
        s4 = f(t + dt, y + dt * s3, *args)
        next_y = y + dt/6 * (s1 + 2*s2 + 2*s3 + s4)
    else:
        s1 = f(t, y, *args)
        s2 = f(
            t + dt/2, 
            [y_i + dt/2 * s1_i for y_i, s1_i in zip(y, s1)], 
            *args)
        s3 = f(
            t + dt/2, 
            [y_i + dt/2 * s2_i for y_i, s2_i in zip(y, s2)], 
            *args)
        s4 = f(
            t + dt, 
            [y_i + dt * s3_i for y_i, s3_i in zip(y, s3)], 
            *args)
        next_y = [
            y + dt/6 * (s1_i + 2*s2_i + 2*s3_i + s4_i) 
                for s1_i, s2_i, s3_i, s4_i in zip(s1, s2, s3, s4)
        ]
    return next_y

def RK_array_explicit(a: List[List[Number]], b: List[Number], c: List[Number]) -> Callable[[Func[T, Y], T, T, Y, Any], Y]:
    """Create a function that generates the next y with given k stages Runge-Kutta coefficients:
    c | a
    -------
      | b^T
    or written explicitly:
    c_0     | a_{00}
    c_1     | a_{10} a_{11}
    ...
    c_{k-2} | a_{k-2, 0} ... a_{k-2, k-2}
    ------------------------------------
            | b_0 ........ b_{k-2}   b_{k-1}
    so that
    y_next = y + (b_0 * s_0 + ... + b_{k-1} * s_{k-1}) * dt
    where
    s_0 = f(t, y)
    s_1 = f(t + c_0 dt, y + (a_{00} s_0) * dt)
    s_2 = f(t + c_1 dt, y + (a_{10} s_0 + a_{11} s_1) * dt)
    ...
    s_{k-1} = f(t + c_{k-2} dt, y + (a_{k-2, 0} s_0 + ... + a_{k-2, k-2} s_{k-2}) * dt)
    Args:
        a (List[List[Number]]): 
        [
            [a_{00}], 
            [a_{10}, a_{11}],
            ...
            [a_{k-2, 0}, ..., a_{k-2, k-2}]
        ]
        b (List[Number]): [b_0, ..., b_{k-2}, b_{k-1}]
        c (List[Number]): [c_0, ..., c_{k-2}]

    Returns:
        Callable[[Func[T, Y], T, T, Y, Any], Y]: func (f, dt, t, y, *args) -> next_y

    Reference: 
    线性方程数值解法 (第二版) by 余德浩，汤华中
    """
    def next_y_RK(f: Func[T, Y], dt: T, t: T, y: Y, *args) -> Y:
        if isinstance(y, Number):
            s = [f(t, y, *args)]
            for a_i, c_i in zip(a, c):
                next_s = f(
                    t + c_i * dt, 
                    y + sum(a_ij * s_i for a_ij, s_i in zip(a_i, s)) * dt,
                    *args
                    )
                s.append(next_s)
            next_y = y + sum(s_i * b_i for b_i, s_i in zip(b, s)) * dt
        else:
            s = [f(t, y, *args)]
            for a_i, c_i in zip(a, c):
                y_stage_i = [
                    y_k + sum(a_ij * s_i[k] for a_ij, s_i in zip(a_i, s)) * dt 
                    for k, y_k in enumerate(y)]
                next_s = f(t + c_i * dt, y_stage_i, *args)
                s.append(next_s)
            next_y = [y_k + sum(s_i[k] * b_i for b_i, s_i in zip(b, s)) * dt for k, y_k in enumerate(y)]
        return next_y
    return next_y_RK

SUPPORTED_METHODS = [
    "Euler",
    "Midpoint",
    "Trapezoid",
    "RK4"
]

_SUPPORTED_IVP_CONST_STEP_METHODS: Dict[str, Callable[[Func[T, Y], T, T, Y, Any], Y]] = {
    "euler": _next_y_Euler,
    "midpoint": _next_y_midpoint,
    "mid-point": _next_y_midpoint,
    "trapezoid": _next_y_trapezoid,
    "rk4": _next_y_RK4,
    "runge-kutta4": _next_y_RK4,
    "runge-kutta-4": _next_y_RK4,
    "rk2": _next_y_trapezoid
}


