"""Basic numerical polynomial operations"""
from itertools import islice

class Polynomial:
    """Polynomial class, with basic operations"""
    def __init__(self, factors: list):
        """generates a polynomial that satisfies y = factors[0] + x * factors[1] + ... + x**d * factors[d]

        Args:
            factors (list): the factors of the polynomial, factors[i] is the factor of x ** i.
        """
        for i in range(len(factors) - 1, -1, -1):
            if factors[i] != 0:
                self.d = i
                self.factors = factors[:i+1]
                break
        else:
            self.d = 0
            self.factors = [0.]

    def __call__(self, x):
        """Using Qin Jiushao (秦九韶) method to evaluate the polynomial's value at x"""
        factors = self.factors
        y = factors[-1]
        for f in islice(reversed(factors), 1, None):
            y = x * y + f
        return y

    def __add__(self, P):
        factors = (self.factors).copy()
        if not isinstance(P, Polynomial):
            factors[0] += P
            return Polynomial(factors)
        else:
            for i in range(min(len(factors), len(P.factors))):
                factors[i] += P.factors[i]
            if P.d > self.d:
                factors += P.factors[self.d+1:]
            return Polynomial(factors)

    def __radd__(self, P):
        return self + P

    def __mul__(self, P):
        if not isinstance(P, Polynomial):
            return Polynomial([P * factor for factor in self.factors])
        else:
            d = self.d + P.d
            factors = []
            for i in range(d + 1):
                factor = 0
                for j in range(max(0, i - P.d), min(i, self.d) + 1):
                    factor += self.factors[j] * P.factors[i-j] 
                factors.append(factor)
            
            return Polynomial(factors)

    def __rmul__(self, P):
        return self * P

    def __repr__(self):
        if self.factors[-1] == 0:
            return "zero Polynomial\n0"

        factors = self.factors
        for i, f in enumerate(factors):
            if f != 0:
                fst_idx_nonzero = i
                break

        poly_str = "{}".format(factors[fst_idx_nonzero])
        if fst_idx_nonzero != 0:
            poly_str += " X"
            if fst_idx_nonzero != 1:
                poly_str += "^{}".format(fst_idx_nonzero)

        for i, f in islice(enumerate(factors), fst_idx_nonzero + 1, None):
            if i == 0 or f == 0:
                continue
            if f < 0:
                sign = " - "
            else:
                sign = " + "
            poly_str += sign
            if i == 1:
                poly_str += "{} X".format(abs(f))
            else:
                poly_str += "{} X^{}".format(abs(f), i)

        return "Polynomial of degree {} \n".format(self.d) + poly_str

    def __truediv__(self, value):
        return self * value**(-1)

    def diff(self, n: int=1):
        d = self.d
        if n > d:
            return zero_poly(type(self.factors[0]))
        
        factors = self.factors
        diff_factors = [0] * (d - n + 1)
        for i in range(d - n + 1):
            diff_factors[i] = factors[i + n]
            for j in range(n):
                diff_factors[i] *= (i + n - j)
        
        return Polynomial(diff_factors)

def zero_poly(type=int):
    return Polynomial([type(0)])
        
        

