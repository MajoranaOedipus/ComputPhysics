from numbers import Number, Real
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

T = TypeVar("T", bound=Real)
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
    N: int = 100, endpoint = True, args: Tuple[Any] = ()) -> List[Y]:
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
        List[Y]: A list of values of the solved function at t_i := t_0 + i * stepsize, for i in (if endpoint then N + 1 else N)
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

    if endpoint:
        N += 1

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
        next_y = [y_i + dt * f(t, y, *args)[i] for i, y_i in enumerate(y)]

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
            y_i + dt/6 * (s1_i + 2*s2_i + 2*s3_i + s4_i) 
                for y_i, s1_i, s2_i, s3_i, s4_i in zip(y, s1, s2, s3, s4)
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

def solve_IVP_RK23(
    f: Func[T, Y], y0: Y = 0, bounds: Tuple[T, T] = (0, 1),
    dt: T = 1e-3, TOL: Y = 1e-6, dt_max: Optional[T] = None,
    args: Tuple[Any] = ()) -> Tuple[List[T], List[Y]]:
    """Solve the IVP ODE problem y' = f(t, y) with initial condition y(t_0) = y_0
        using RK2/3 (embedded RK pair).

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
        args (Tuple[Any], optional): Additional args to be passed to f. Defaults to ().

    Returns:
        Tuple[List[T], List[Y]]: (ts, ys) 
    """
    y = y0
    t, t_end = bounds
    retry = False   # is it the second time to rechoose dt?
    y_output = [y0]
    t_output = [t]
    if isinstance(y, Number):
        while t < t_end:
            s1 = f(t, y, *args)
            s2 = f(t + dt, y + s1 * dt, *args)
            s3 = f(t + dt/2, y + 1/4 * (s1 + s2) * dt, *args)
            if y:
                error_y_rel = abs(dt/3 * (s1 - 2 * s3 + s2)) / y    # |next_y(RK3) - next_y(RK2)| / y
            else:
                error_y_rel = abs(dt/3 * (s1 - 2 * s3 + s2))    # |next_y(RK3) - next_y(RK2)|

            if error_y_rel < TOL:
                if t + dt > t_end:
                    dt = t_end - t
                    continue
                t += dt
                t_output.append(t)
                y += dt/6 * (s1 + s2 + 4*s3)    # RK3
                dt = _next_dt(dt, error_y_rel, 2, TOL, dt_max)
                y_output.append(y)
                retry = False
            else:
                if retry:
                    dt /= 2
                else:
                    dt = _next_dt(dt, error_y_rel, 2, TOL, dt_max)
                    retry = True
    else:
        while t < t_end:
            s1 = f(t, y, *args)
            s2 = f(t + dt, [y_i + s1[i] * dt for i, y_i in enumerate(y)], *args)
            s3 = f(t + dt/2, [y_i + 1/4 * (s1[i] + s2[i]) * dt for i, y_i in enumerate(y)], *args)
            error_y = dt/3 * sum(
                (s1_i - 2 * s3_i + s2_i) ** 2 
                    for s1_i, s2_i, s3_i in zip(s1, s2, s3)) ** 0.5    # |next_y(RK3) - next_y(RK2)|
            if y:
                error_y_rel =  error_y / sum(y_i**2 for y_i in y) ** 0.5    # |next_y(RK3) - next_y(RK2)| / y
            else:
                error_y_rel = error_y

            if error_y_rel < TOL:
                if t + dt > t_end:
                    dt = t_end - t
                    continue
                t += dt
                t_output.append(t)
                next_y = []
                for i, y_i in enumerate(y):
                    next_y.append(y_i +  dt/6 * (s1[i] + s2[i] + 4*s3[i]))   # RK3
                y_output.append(next_y)
                dt = _next_dt(dt, error_y_rel, 2, TOL, dt_max)
                y = next_y
                retry = False
            else:
                if retry:
                    dt /= 2
                else:
                    dt = _next_dt(dt, error_y_rel, 2, TOL, dt_max)
                    retry = True
        
    return t_output, y_output

def solve_IVP_RKF45(
    f: Func[T, Y], y0: Y = 0, bounds: Tuple[T, T] = (0, 1),
    dt: T = 1e-3, TOL: Y = 1e-6, dt_max: Optional[T] = None,
    args: Tuple[Any] = ()) -> Tuple[List[T], List[Y]]:
    """Solve the IVP ODE problem y' = f(t, y) with initial condition y(t_0) = y_0
        using RKF4/5 (embedded RK pair).

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
        args (Tuple[Any], optional): Additional args to be passed to f. Defaults to ().

    Returns:
        Tuple[List[T], List[Y]]: (ts, ys) 
    """
    y = y0
    t, t_end = bounds
    retry = False   # is it the second time to rechoose dt?
    y_output = [y0]
    t_output = [t]
    if isinstance(y, Number):
        while t < t_end:
            s1 = f(t, y, *args)
            s2 = f(t + dt/4, y + s1 * dt/4, *args)
            s3 = f(t + 3*dt/8, y + dt/32 * (3 * s1 + 9 * s2), *args)
            s4 = f(t + 12/13 * dt, y + dt/2197 * (1932 * s1 - 7200 * s2 + 7296 * s3), *args)
            s5 = f(t + dt, y + dt * (439/216 * s1 - 8 * s2 + 3680/513 * s3 - 845/4104 * s4), *args)
            s6 = f(
                t + dt/2, 
                y + dt * (-8/27 * s1 + 2 * s2 - 3544/2565 * s3 + 1859/4104 * s4 - 11/40 * s5),
                *args)
            error_y = abs(dt * (s1/360 - 128/4275 * s3 
                                - 2197/75_240 * s4 + s5 / 50 + 2/55 * s6))  # |next_y(RK5) - next_y(RK4)|
            if y:
                error_y_rel = error_y / y    # |next_y(RK5) - next_y(RK4)| / y
            else:
                error_y_rel = error_y 

            if error_y_rel < TOL:
                if t + dt > t_end:
                    dt = t_end - t
                    continue
                t += dt
                t_output.append(t)
                y += dt * (16/135 * s1 + 6656/12_825 * s3 + 28_561/56_430 * s4 - 9/50 * s5 + 2/55 * s6)    # RK5
                dt = _next_dt(dt, error_y_rel, 4, TOL, dt_max)
                y_output.append(y)
                retry = False
            else:
                if retry:
                    dt /= 2
                else:
                    dt = _next_dt(dt, error_y_rel, 4, TOL, dt_max)
                    retry = True
    else:
        while t < t_end:
            s1 = f(t, y, *args)
            s2 = f(
                t + dt/4, 
                [y_i + s1[i] * dt/4 for i, y_i in enumerate(y)], 
                *args)
            s3 = f(
                t + 3*dt/8, 
                [y_i + dt/32 * (3 * s1[i] + 9 * s2[i]) for i, y_i in enumerate(y)], 
                *args)
            s4 = f(
                t + 12/13 * dt, 
                [y_i + dt/2197 * (1932 * s1[i] - 7200 * s2[i] + 7296 * s3[i]) 
                    for i, y_i in enumerate(y)], 
                *args)
            s5 = f(
                t + dt, 
                [y_i + dt * (439/216 * s1[i] - 8 * s2[i] + 3680/513 * s3[i] - 845/4104 * s4[i])
                    for i, y_i in enumerate(y)], 
                *args)
            s6 = f(
                t + dt/2, 
                [y_i + dt * (-8/27 * s1[i] + 2 * s2[i] - 3544/2565 * s3[i] + 1859/4104 * s4[i] - 11/40 * s5[i])
                    for i, y_i in enumerate(y)],
                *args)
            error_y = dt * sum(
                (s1_i/360 - 128/4275 * s3_i
                    - 2197/75_240 * s4_i + s5_i / 50 + 2/55 * s6_i)**2 
                        for s1_i, s3_i, s4_i, s5_i, s6_i 
                            in zip(s1, s3, s4, s5, s6)) ** 0.5  # |next_y(RK5) - next_y(RK4)|
            y_length = sum(y_i**2 for y_i in y) ** 0.5
            if y_length:
                error_y_rel = error_y / sum(y_i**2 for y_i in y) ** 0.5    # |next_y(RK5) - next_y(RK4)| / y
            else:
                error_y_rel = error_y   

            if error_y_rel < TOL:
                if t + dt > t_end:
                    dt = t_end - t
                    continue
                t += dt
                t_output.append(t)
                
                next_y = []
                for i, y_i in enumerate(y):
                    next_y.append(y_i + dt * (16/135 * s1[i] + 6656/12_825 * s3[i] 
                                    + 28_561/56_430 * s4[i] - 9/50 * s5[i] + 2/55 * s6[i]))    # RK5
                dt = _next_dt(dt, error_y_rel, 4, TOL, dt_max)
                y_output.append(next_y)
                y = next_y
                retry = False
            else:
                if retry:
                    dt /= 2
                else:
                    dt = _next_dt(dt, error_y_rel, 4, TOL, dt_max)
                    retry = True
        
    return t_output, y_output

def _next_dt(dt: T, error_rel: Y, p: int, TOL: Y, dt_max: Optional[T] = None) -> T:
    """Generate next dt in RK embedded pair

    Args:
        dt (T): The last dt
        error_rel (Y): estimated relative error
        p (int): the order of the solver
        TOL (T): tolerance

    Returns:
        T: next dt
    """
    if error_rel:
        K = 0.8 * (TOL /error_rel)**(1/(p + 1))
    else:
        K = 2
    if dt_max is None:
        return K * dt
    else:
        return min((dt_max, K * dt))

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


