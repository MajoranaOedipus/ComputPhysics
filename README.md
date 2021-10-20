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
from ComputPhysics.Interpolation import interpolate_Lagrange
interpolate_Lagrange([3, 1, 2], [1, 2, 3])
```
Output:
```
Polynomial of degree 2 
-2.0 + 5.5 X - 1.5 X^2
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
* More generic (**works required...**)

### Interpolation
* Lagrange interpolation (*finished*)
* Hermit interpolation (**WIP**)
* Spline interpolation (**WIP**)
* Maybe a class for result? (**WIP**)

### Numerical Differentiation
* differentiation using central difference (*finished*)
* adaptive steps (**WIP**)

### Numerical Integration
WIP
### FFT
WIP
### ODE
WIP
### Basic Probability and Statistic
WIP
### Monte Carlo Methods
WIP
### ...
