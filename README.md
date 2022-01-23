# ComputPhysics
Pure Python Computational Physics Package (WIP, For Study)

**Use PyPy for a better perfomance!** 

Clone this repository add it to your `PATH` or `cd` to `ComputPhysics/..` and try the following in your PyPy REPL or IPython:

## Examples
### Matrices

```python
from ComputPhysics.LinearAlgebra import Matrix, eye
A = Matrix(list(range(9)), (3, 3)) + eye(3)
A_inv = A.inverse()
A * A_inv - eye(3)
```
Output: 
```(3,3) Matrix
[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.220446049250313e-16, -4.440892098500626e-16]]
```

### System of Linear Equations
```python
from ComputPhysics.LinearAlgebra import Matrix, solve_linear
A = Matrix([
    [1, 2, 3, 1],
    [1, 3, 4, 2],
    [2, 7, 9, 3],
    [3, 7, 10, 2]
])
b = Matrix([3, 2, 7, 12], (4, 1))
sols = solve_linear(A, b)
sols
```
Output: 
```
{'nonzero_sols_homo': [(4,1) Matrix
  [[-1.0], [-1.0], [1.0], [-0.0]]],
 'sol_inhomo': (4,1) Matrix
 [[3.0], [1.0], [0.0], [-2.0]],
 'solable': True}
```


### Numerical Differentiation

```python
from ComputPhysics.Differentiation import diff_f
f = lambda x: x**3
f_p = diff_f(f)
f_pp = diff_f(f, n=2)
f(1), f_p(1), f_pp(1)
```
Output: `(1, 4.0, 6.0)`

### Polynomial
```python
from ComputPhysics.Polynomial import Polynomial
P = Polynomial([1, 2, 3])
P, "P(1) = {}".format(P(1))
```
Output:
```
(Polynomial of degree 2 
 1 + 2 X + 3 X^2,
 'P(1) = 6')
```

### Interpolation
```python
from ComputPhysics.Interpolation import interpolate_Lagrange, interpolate_Hermite
print(interpolate_Lagrange([3, 1, 2], [1, 2, 3]),
      interpolate_Hermite([0, 1], [[1, 0], [2, 2]]), sep="\n")
```
Output:
```
Polynomial of degree 2 
-2.0 + 5.5 X - 1.5 X^2
Polynomial of degree 2 
1.0 + 1.0 X^2
```

## Integration
```python
from ComputPhysics.Integration import SUPPORTED_METHOD, quad

def func(x):
    return (x)**3

N = 10
for method in SUPPORTED_METHOD:
    print(method+":", quad(func, 0, 1, method=method))
```
Output:
```
Romberg: 0.25
Newton-Cotes: 0.2500000000000001
midpoint: 0.2487375000000001
trapezoid: 0.25250000000000006
Simpson: 0.25
```

## Root-finding

```python
from ComputPhysics.Optimisation import *
Nmax = 10
x0_bin = solve_binary(cos, 1, 2, Nmax=Nmax)
x0_iter = solve_iter(cos, 1, Nmax=Nmax)
x0_Newton = solve_Newton(cos, 1, Nmax=Nmax)
x0_secant = solve_secant(cos, 0, 1, Nmax=Nmax)
x0_regula_falsi = solve_regula_falsi(cos, 0, 1, Nmax=Nmax)

print(f"bin          : x_0 = {x0_bin}, where f(x_0) = {cos(x0_bin)}")
print(f"iter         : x_0 = {x0_iter}, where f(x_0) = {cos(x0_iter)}")
print(f"Newton       : x_0 = {x0_Newton}, where f(x_0) = {cos(x0_Newton)}")
print(f"secant       : x_0 = {x0_secant}, where f(x_0) = {cos(x0_secant)}")
print(f"regula falsi : x_0 = {x0_regula_falsi}, where f(x_0) = {cos(x0_regula_falsi)}")
```

Output:
```
bin          : x_0 = 1.5712890625, where f(x_0) = -0.0004927356851649559
iter         : x_0 = 1.5707963267948966, where f(x_0) = 6.123233995736766e-17
Newton       : x_0 = 1.5707963267948966, where f(x_0) = 6.123233995736766e-17
secant       : x_0 = 1.5707963267948966, where f(x_0) = 6.123233995736766e-17
regula falsi : x_0 = 1.5707963267948966, where f(x_0) = 6.123233995736766e-17
```

## ODE
```Python
from ComputPhysics.ODE import *
def f(t, y):
    return y + t

def y(t):
    return exp(t) - t - 1

y0 = 0
N = 10
y_solve_Euler = solve_IVP_explicit(f, y0, N=N, method="Euler")
y_solve_trap = solve_IVP_explicit(f, y0, N=N, method="trapezoid")
y_solve_mid = solve_IVP_explicit(f, y0, N=N, method="midpoint")
y_solve_RK4 = solve_IVP_explicit(f, y0, N=N, method="RK4")
y_solve_RK4_2 = solve_IVP_explicit(f, y0, N=N, method=RK_array_explicit(
    [
        [1/2], 
        [0, 1/2], 
        [0, 0, 1]
    ], 
    [1/6, 1/3, 1/3, 1/6], 
    [1/2, 1/2, 1]
    )) 

y_exact = [y(i/N) for i in range(N)]
difference_Euler = (sum((y1-y2)**2 for y1, y2 in zip(y_solve_Euler, y_exact)) / N) ** 0.5
difference_trap = (sum((y1-y2)**2 for y1, y2 in zip(y_solve_trap, y_exact)) / N) ** 0.5
difference_mid = (sum((y1-y2)**2 for y1, y2 in zip(y_solve_mid, y_exact)) / N) ** 0.5
difference_RK4 = (sum((y1-y2)**2 for y1, y2 in zip(y_solve_RK4, y_exact)) / N) ** 0.5
difference_RK4_2 = (sum((y1-y2)**2 for y1, y2 in zip(y_solve_RK4_2, y_exact)) / N) ** 0.5
```

Output:
```
Euler: 51.552760906E-3
trap : 1.729840013E-3
mid  : 1.729840013E-3
RK4  : 0.000858108E-3
RK4-2: 0.000858108E-3
```

```Python
from ComputPhysics.ODE import solve_IVP_RK23, solve_IVP_RKF45
from math import sin

def g(t, ys):
    y1, y2 = ys
    return [y2 * sin(t), -10]

ys_0 = [0, 10]
bounds = [0, 4]
ts_RK23, ys_RK23 = solve_IVP_RK23(g, ys_0, bounds=bounds, TOL=1e-6)
ts_RKF45, ys_RKF45 = solve_IVP_RKF45(g, ys_0, bounds=bounds, TOL=1e-6)
len(ts_RK23), ts_RK45, ys_RK45
```

Output:
```
(15,
 [0, 0.001, 2.100545200327771, 4.0],
 [[0, 10],
  [4.996666250333349e-06, 9.99],
  [-4.116118414748008, -11.005452003277712],
  [-1.9838284731989457, -30.0]])

```

## Projects
### Linear Algebra
Numerical matrix manipulation and linear systems
* Basic matrices creation and operation (*finished*)
* Determinent (*finished*) 
* Inverse and any-integer power (*finished*)
* System of Linear Equations (*finished*, more testing required)
* SVD (**WIP**)
* Matrix functions (exp, sin, etc., **WIP**)
* More generic (**works required...**)
* ...

### Polynomials
Tool packages for other packages
* Polynomial operations (*finished*, `__mul__` needs optimation by FFT)
* Called as a function (*finished*)
* Euclidean division, modulus (**WIP**)
* Find roots (**WIP**)

### Interpolation
* Lagrange interpolation (*finished*)
* Hermit interpolation (*finished*)
* Spline interpolation (**WIP**)

### Numerical Differentiation
* Differentiation using central difference (*finished*)
* Adaptive steps (**WIP**)

### Numerical Integration
* Now supported method: Romberg, Newton-Cotes, midpoint, trapezoid, Simpson (*finished*)
* User interface `quad` (*finished*)
* Adaptive steps (**WIP**)

### ODE
#### IVP:
* Euler (eplicit) method (*finished*)
* Midpoint, trapzoid (*finished*)
* RK4 (*finished*)
* RK2/3, RKF4/5 (*finished*)
* ...

### PDE
WIP

### Optimisation
#### root finding
* binary search (*finished*)
* Newton's method (*finished*)
* Secant method (*finished*)
* regula falsi (*finished*)
* Muller, IQI and Brent ... (**WIP**)
#### local optimization (**WIP**)
* ...
### FFT
WIP
### Basic Probability and Statistic
WIP
### Monte Carlo Methods
WIP
### ...
