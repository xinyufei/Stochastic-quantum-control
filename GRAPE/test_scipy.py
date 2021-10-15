import numpy as np
import scipy.optimize
from scipy.optimize import Bounds
from scipy.optimize import minimize
from scipy.optimize import fmin_l_bfgs_b

L = 1000
c = [0.01*(l+1) for l in range(L)]
p = [1/L] * L
eta = 0.01


def obj(*args):
    zeta = args[0]
    obj = zeta + 1 / eta * sum(p[l] * max([0, c[l] - zeta]) for l in range(L))
    return obj


def fprime(*args):
    zeta = args[0]
    g = 1.0
    for l in range(L):
        if zeta <= c[l]:
            g -= p[l] / eta
    return g


def test_flbfgsb():
    min_grad = 1e-6
    max_iter_step = 100
    results = fmin_l_bfgs_b(obj, np.array([0.0]), bounds=None, pgtol=min_grad,
                            fprime=fprime, maxiter=max_iter_step)
    # minimize the objective function
    # bounds = Bounds([-np.infty], [np.infty])
    # results = minimize(obj, [0], method='L-BFGS-B', bounds=bounds, jac=fprime, options={'ftol': 1e-06})
    print(results)


if __name__ == '__main__':
    test_flbfgsb()
