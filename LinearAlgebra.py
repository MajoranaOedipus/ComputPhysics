from ._util import _flatten
from ._const import EPS
from operator import add, mul


class Matrix:
    def __init__(self, elements, shape=None):
        if type(elements) == Matrix:
            self.elements = elements.elements
            self.shape = shape
            return
        if shape != None: 
            try:
                n, m = shape
            except TypeError:
                n = shape
                m = 1
            elements = _flatten(elements)
            if len(elements) != n * m:
                raise ValueError("The number of the elements cannot fit the shape of the matrix")
            self.elements = [elements[i*m:(i+1)*m] for i in range(n)]
        else:
            n = len(elements)
            try:
                m = len(elements[0])
            except TypeError:
                self.shape = (1, n)
                self.elements = [elements]
                return
            
            for row in elements:
                if len(row) != m:
                    raise ValueError("The shape of a matrix shall be rectangular. ")
            self.elements = elements

        self.shape = (n, m)
        return

    def __getitem__(self, i):
        try:
            return self.elements[i[0]][i[1]]
        except TypeError:
            return self.elements[i]

    def __setitem__(self, i, item):
        try:
            self.elements[i[0]][i[1]] = item
        except TypeError:
            self.elements[i] = item

    def is_diag(self, error: float=EPS) -> bool:
        """Check if a matrix is a diagonal square matrix

        Args:
            error (float, optional): Allowed error. Defaults to EPS i.e. the machine error.

        Returns:
            bool: if the matrix is square and diagonal, returns True, elsewise False.
        """
        n, m = self.shape
        if n != m:
            return False
        for i, j in zip(range(n), range(m)):
            if abs(self[i, j]) < error and i != j:
                return False
        else:
            return True

    def __float__(self) -> float:
        """Convert a $\lambda I_n$-like matrix to a floating number.

        Returns:
            float: the diagonal value $\lambda$.
        """
        n, m = self.shape
        if n == m:
            if self.is_diag():
                diag = self[0, 0]
                for i in range(n):
                    if self[i, i] != diag:
                        raise ValueError("Cannot convert a matrix that has different diagonal values into a float.")
                else:
                    return float(diag)
            else:
                raise ValueError("Cannot convert a non-diagonal matrix into a float.")
        else:
            raise ValueError("Cannot convert a non-square matrix into a float.")

    def __str__(self):
        return str(self.elements)
    
    def __iter__(self):
        return iter(self.elements)

    def __add__(self, B):
        A = self.elements
        n = self.shape[0]
        if type(B) == Matrix:
            if self.shape != B.shape:
                raise ValueError("The shape of two matrices shall be the same.")
            C = [list(map(add, A[i], B[i])) for i in range(n)]
        else:
            C = [[a + B for a in A[i]] for i in range(n)]
        return Matrix(C)
    
    def __radd__(self, B):
        return self + B
    
    def __sub__(self, B):
        return self + (-1.) * B

    def __rsub__(self, B):
        return B + (-1.) * self

    def __repr__(self):
        return "({},{}) Matrix\n{}".format(*self.shape, str(self))

    def __eq__(self, B):
        return type(B) == Matrix and B.elements == self.elements

    def T(self):
        return Matrix([list(pair) for pair in zip(*self.elements)])

    def __mul__(self, B):
        A = self.elements
        nA, mA = self.shape
        if type(B) == Matrix:
            nB, mB = B.shape
            if mA != nB:
                raise ValueError("In order for A * B to make sense, the number of columns of A must be equal to the number of rows of B.")
            C = [[sum(map(mul, A[i], B.T()[j])) for j in range(mB)] for i in range(nA)]
        else:
            C = [list(map(lambda x: x * B, A[i])) for i in range(nA)]
        return Matrix(C)

    def __rmul__(self, B):
        return self * B

    def det(self) -> float:
        n, m = self.shape
        if n != m:
            raise ValueError("Cannot compute the determinent for non-square matrix.")
        A = [row.copy() for row in self.elements] # Copy the matrix's elements
        for j in range(n - 1):
            for k in range(j, n): # find the first row that is not zero
                if A[k][j] != 0:
                    break
            else:
                return 0. # A matrix with a zero column has a 0 determinent
            if k != j:
                for l in range(j, n):
                    A[j][l] += A[k][l]
                
            for i in range(j + 1, n):
                k = A[i][j] / A[j][j]
                for l in range(j, n):
                    A[i][l] -= k * A[j][l]
        result = 1.
        for i in range(n):
            result *= A[i][i] 
        return result

    def tr(self) -> float:
        n, m = self.shape
        if n != m:
            raise ValueError("Cannot compute the trace for non-square matrix.")
        trace = 0.
        for i in range(n):
            trace += self[i, i]
        return trace

    def triangularize(self, pos="upper", bounds=None): # TODO: reimplement triangularization about bounds, make changes outside the bounds
        if bounds == None:
            row_l, col_l = 0, 0
            row_u, col_u = self.shape
        else:
            row_l, col_l = bounds[0]
            row_u, col_u = bounds[1]
            n, m = self.shape
            if not (0 <= row_l < row_u <= n) or not (0 <= col_l < col_u <= m):
                raise ValueError ("Bounds out of range")

        A = self.elements
        if pos == "upper":
            j_range = range(row_l, row_u - 1)
            j_row = lambda j: j
            k_range_arg = lambda j: (j, row_u)
            i_range_arg = lambda j: (j + 1, row_u)
            l_range_arg = lambda j: (j, col_u)
        elif pos == "lower":
            j_range = range(col_u - 1, col_u - row_u, -1)
            j_row = lambda j: row_u + j - col_u
            k_range_arg = lambda j: (row_u + j - col_u, row_l - 1, -1)
            i_range_arg = lambda j: (row_u + j - col_u - 1, row_l - 1, -1)
            l_range_arg = lambda j: (col_l, j + 1)
        else:
            raise ValueError("'pos' must be 'upper' or 'lower'")

        for j in j_range:
            zero_column = False
            for k in range(*k_range_arg(j)): # find the first row that is not zero
                if A[k][j] != 0:
                    break
            else:
                zero_column = True
            if k != j_row(j):
                for l in range(*l_range_arg(j)):
                    A[j_row(j)][l] += A[k][l]
                
            if not zero_column:
                for i in range(*i_range_arg(j)):
                    K = A[i][j] / A[j_row(j)][j]
                    for l in range(*l_range_arg(j)):
                        A[i][l] -= K * A[j_row(j)][l]

    def is_inversible(self):
        return self.det() != 0

    def T(self):
        return Matrix([list(row) for row in zip(*self)])

    def inverse(self): 
        n, m = self.shape
        if n != m:
            raise ValueError ("Cannot compute inverse of a non-square matrix")
        I_n = eye(n)
        A = concatenate(self, I_n)
        A.triangularize()
        for j in range(n):
            eigenvalue = A[j, j]
            if eigenvalue == 0.:
                raise ValueError("The matrix is not inversible.")
            for l in range(j, 2*n):
                A[j, l] /= eigenvalue
            for i in range(j):
                K = A[i, j]
                for l in range(j, 2*n):
                    A[i, l] -= K * A[j, l]
        inversed = Matrix([row[n:] for row in A])
        return inversed
    
    def __pow__(self, p):
        n, m = self.shape
        if n != m:
            raise ValueError ("Cannot compute power for non-square matrix")
        
        if p > 0:
            A = self
        else:
            A = self.inverse()
            p = -p
        result = eye(n)
        if p < 16:
            for _ in range(1, p+1):
                result *= A
        else:
            gen_bin = (int(i) for i in bin(p)[2:])
            A2i = A
            for bi in gen_bin:
                if bi == 1:
                    result *= A2i
                A2i **= 2 
        return result

def concatenate(A, B, vertical=False):
    n_A, m_A = A.shape
    n_B, m_B = A.shape
    if not vertical:
        if n_A != n_B:
            raise ValueError ("Cannot concatenate two matrices with different number of rows")
        A = A.elements
        B = B.elements
        C = [A[i] + B[i] for i in range(n_B)]
        return Matrix(C)
    else:
        if m_A != m_B:
            raise ValueError ("Cannot stack two matrices with different number of columns")
        A = A.elements
        B = B.elements
        C = A + B
        return Matrix(C)

def triangularize(A, pos="upper", bounds=None): 
    A = Matrix([row.copy() for row in A.elements]) # Copy the matrix's elements
    A.triangularize(pos, bounds)
    return A
        
def eye(n):
    O_n = zeros(n, n)
    for i in range(n):
        O_n[i, i] = 1.
    I_n = O_n
    return I_n

def zeros(n, m=1):
    O_n = [[0.]* m for _ in range(n)]
    return Matrix(O_n)

def timeit(f, N=10000):
    from time import time
    t0 = time()
    for _ in range(N):
        f()
    dt = time() - t0
    print("total cost: {} ms, per loop: {} us".format(dt*1e3, dt/N * 1e6))

def matrixify(func, *args, **kwargs):
    def Mfunc(A):
        if isinstance(A, Matrix):
            n, _ = A.shape
            A_func = [[func(a, *args, **kwargs) for a in A[i]] for i in range(n)]
            return Matrix(A_func)
        else:
            return func(A)
    return Mfunc 