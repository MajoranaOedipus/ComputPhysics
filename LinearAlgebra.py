from operator import add, mul
eps = abs(7./3 - 4./3 -1) # Machine error

def _flatten(a: list):
    a_flatten = []
    for sublist in a:
        try:
            a_flatten += sublist
        except TypeError:
            a_flatten.append(sublist)
    return a_flatten

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

    def is_diag(self, error=eps) -> bool:
        n, m = self.shape
        if n != m:
            return False
        for i, j in zip(range(n), range(m)):
            if abs(self[i, j]) < error and i != j:
                return False
        else:
            return True

    def __float__(self):
        """
        Convert a $\lambda I_n$-like matrix to a floating number.
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
            C = [list(map(lambda x: x + B, A[i])) for i in range(n)]
        return Matrix(C)
    
    def __radd__(self, B):
        return self + B

    def __repr__(self):
        return "({},{}) Matrix\n{}".format(*self.shape, str(self))

    def __eq__(self, B):
        return type(B) == Matrix and B.elements == self.elements

    def T(self):
        return Matrix([list(pair) for pair in zip(*self.elements)])

    def __mul__(self, B) -> Matrix: # TODO: make it faster
        A = self.elements
        nA, mA = self.shape
        if type(B) == Matrix:
            nB, mB = B.shape
            if mA != nB:
                raise ValueError("In order for A * B to make sense, the number of columns of A must be equal to the number of rows of the B.")
            C = [[sum(map(mul, A[i], B.T()[j])) for j in range(mB)] for i in range(nA)]
        else:
            C = [list(map(lambda x: x * B, A[i])) for i in range(nA)]
        return Matrix(C)

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
                A[j][j:] = [A[j][l] + A[k][l] for l in range(j, n)]
                
            for i in range(j + 1, n):
                k = A[i][j] / A[j][j]
                A[i][j:] = [A[i][l] - k * A[j][l] for l in range(j, n)]
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

    def is_inversible(self):
        return self.det() != 0

    def T(self):
        return Matrix([list(row) for row in zip(*self)])

    def inverse(self): # TODO: implement this

        return None
        

def timeit(f, N=10000):
    from time import time
    t0 = time()
    for _ in range(N):
        f()
    dt = time() - t0
    print("total cost: {} ms, per loop: {} us".format(dt*1e3, dt/N * 1e6))

if __name__ == "__main__":
    from random import random
    n = 10
    X = Matrix([random() for _ in range(n ** 2)], (n, n))
    timeit(X.det, 1000)
    print("OK")