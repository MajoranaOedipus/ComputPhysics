# PyComPhy
Pure Python Computational Physics Package (For Study)

**Use PyPy for a better perfomance!** 

WIP

Clone this repository add it to your `PATH` or `cd` to `PyComPhy/..` and try the following in your PyPy REPL or IPython:

```python
from PyComPhy.Differentiation import diff_f
f = lambda x: x**3
f_p = diff_f(f)
f_pp = diff_f(f, n=2)
f(1), f_p(1), f_pp(1)
```

and

```python
from PyComPhy.LinearAlgebra import Matrix, eye
A = Matrix(list(range(9)), (3, 3)) + eye(3)
A_inv = A.inverse()
A * A_inv - eye(3)
```

## Linear Algebra: Numerical matrix manipulation
* Basic matrices creation and operation (*finished*, needs optimation)
* Determinent (*finished*) 
* SVD (**WIP**)
* Inverse and any-integer power (*finished*)
* Matrix functions (exp, sin, etc., **WIP**)
* ...
## Numerical Differentiation (WIP)
* differentiation using central difference (*finished*)
* adaptive steps (**WIP**)
## Numerical Integration
## ...
