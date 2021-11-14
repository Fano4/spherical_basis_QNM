import numpy as np
from scipy import linalg as lg
import math


def rayleigh_nep_solver(A: callable, x0: np.ndarray, z0: complex) -> list:
    maxit = 20
    h = 1e-2
    convergence = True
    precision = 1e-8

    # Enter optimization
    while convergence:
        print("Entering eigenvalue search")
        norm_Ax = 1
        conv_energy = 1
        x = x0 / lg.norm(x0)
        y = np.conj(x)
        z = z0
        zm1 = 0

        for ite in range(maxit):
            print("#############################")
            print("Iteration", ite)
            mat_z = A(z)
            print("Matrix at point z: ")
            print(mat_z)
            dmat_z = finite_diff(A, z, h, degree=4)
            print("Matrix derivative at point z: ")
            print(dmat_z)
            print("Vector :")
            print(x)
            norm_Ax = lg.norm(np.matmul(mat_z, x))
            print('norm_Ax')
            print(norm_Ax)

            conv_energy = np.abs(z - zm1) / np.abs(z)
            print('conv_energy')
            print(conv_energy)

            vkp1 = lg.solve(np.conj(mat_z), np.matmul(np.conj(dmat_z), y))
            print('vkp1')
            print(vkp1)
            ukp1 = lg.solve(mat_z, np.matmul(dmat_z, x))
            print('ukp1')
            print(ukp1)

            zm1 = z

            xm1 = x
            x = ukp1 / lg.norm(ukp1)
            y = vkp1 / lg.norm(vkp1)
            print('x')
            print(x)
            print('y')
            print(y)

            vec_conv = lg.norm(x - xm1)
            if vec_conv > lg.norm(x + xm1):
                vec_conv = lg.norm(x + xm1)
                x = -x
                z = -z
            print('vec_conv')
            print(vec_conv)

            rhok = np.matmul(y.T, np.matmul(mat_z, x)) / np.matmul(y.T, np.matmul(dmat_z, x))
            print('rhok')
            print(rhok)

            z = z - rhok
            print("Energy : ", z)
            print("vector convergence : ", vec_conv)
            print(x)

            if (norm_Ax < precision and vec_conv < precision) or (conv_energy < precision and vec_conv < precision):
                return [z, x, y]

            elif np.isnan(z):
                raise ValueError("Undefined value of z. Terminating")

        convergence = False
        print("The Nonlinear eigenvalue problem did not converge. ")
        print("Convergence at termination:")
        print("    Energy: ", end='')
        print(norm_Ax)
        print("    Eigenvector norm: ", end='')
        print(conv_energy)
        print("Eigenvalues at termination:")
        print(z)

    return []


def finite_diff(*args, **kwargs):
    A = args[0]
    z = args[1]
    h = args[2]

    if 'order' in kwargs:
        order = kwargs['order']
    else:
        order = 1
    if 'degree' in kwargs:
        degree = kwargs['degree']
    else:
        degree = 2

    coeff = finite_diff_coeff(order, degree)
    n_coeff = len(coeff)
    pmin = - (n_coeff - 1) / 2

    val = np.zeros(A(z).shape, dtype=complex)

    for i in range(n_coeff):
        val = val - coeff[i] * A(z - (pmin + i) * h) / h ** order

    return val


def finite_diff_coeff(order, degree):
    ordmax = 2
    degmax = 8
    if order > ordmax:
        raise ValueError("Maximum derivative order is ", ordmax)
    if degree > degmax:
        raise ValueError("Maximum finite difference degree is ", degmax)
    coeff = [[np.array([-0.5, 0, 0.5]),
              np.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]),
              np.array([-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60]),
              np.array([1 / 280, -4 / 105, 1 / 5, -4 / 5, 0, 4 / 5, -1 / 5, 4 / 105, -1 / 280])],
             [np.array([1, -2, 1]),
              np.array([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]),
              np.array([1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90]),
              np.array([-1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560])]]

    if degree % 2 == 0:
        return coeff[int(order - 1)][int(degree / 2 - 1)]
    else:
        raise ValueError('Invalid degree for finite difference. Only even degree is authorized')
