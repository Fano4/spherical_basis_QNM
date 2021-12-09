from functools import partial
import numpy as np
from scipy import linalg as lg

import plot_functions


def rayleigh_nep_solver(A: callable, x0: np.ndarray, zvec: (complex, list)) -> list:
    maxit = 300
    h_coarse = 1e-4
    h_prec = 1e-4
    h = h_coarse
    maxrad = 0.5
    convergence = True
    precision = 1e-4
    ztab = []
    xtab = []
    ytab = []
    num_eigval = 0
    # Enter optimization
    while convergence:
        print("==================================")
        print("Entering eigenvalue search")
        norm_Ax = 1
        converged = False
        x = x0 / lg.norm(x0)
        y = np.conj(x)

        if isinstance(zvec, list):
            print("Looking for eigenvalue ", num_eigval, " / ", len(zvec))
            z0 = zvec[num_eigval]
        else:
            z0 = zvec

        z = z0
        looping = 0
        bas_fun_index = 0

        for ite in range(maxit * len(x)):
            print("#############################")
            print("Iteration", ite)

            mat_z = A(z)

            if looping > 0:
                looping = looping - 1

            if len(ztab) != 0:
                print("Matrix deflation : ", len(ztab), " eigenvalues")
                mat_z = A(z)
                for i in range(len(ztab)):
                    mat_z = mat_z - 1 / (z - ztab[i] + 1e-15) * np.outer(xtab[i], ytab[i])

            dmat_z = finite_diff(A, z, h, degree=8)
            mem = norm_Ax
            norm_Ax = lg.norm(np.matmul(mat_z, x))
            print('norm_Ax : ', end='')
            print(norm_Ax)
            # if lg.det(mat_z) != 0:
            vkp1 = lg.solve(np.conj(mat_z), np.matmul(np.conj(dmat_z), y))
            ukp1 = lg.solve(mat_z, np.matmul(dmat_z, x))

            xm1 = x
            x = ukp1 / lg.norm(ukp1)
            y = vkp1 / lg.norm(vkp1)

            # vec_conv = lg.norm(x - xm1)
            # if vec_conv > lg.norm(x + xm1) and lg.norm(x + xm1) < 0.1:
            #    print("Inverting eigenvector to reduce phase jump")
            #    print(lg.norm(x - xm1), " > ", lg.norm(x + xm1))
            #    print(xm1)
            #    print("===>")
            #    print(x)
            #    x = -x
            #    y = -y
            rhok = np.matmul(y.T, np.matmul(mat_z, x)) / np.matmul(y.T, np.matmul(dmat_z, x))

            if (abs(norm_Ax - mem) < 1000 and abs(rhok) < 1e-2):
                print("Decrease energy steps near convergence: ", end='')
                print("Initial_energy: ", end='')
                print(z)
                z = z - 0.1 * rhok
                h = h_prec
                print('energy change: ', end='')
                print(-0.1 * rhok)
                print("Energy val :")
                print(z)
            else:
                print("Initial_energy: ", end='')
                print(z)
                h = h_coarse
                z = z - rhok
                print('energy change: ', end='')
                print(-rhok)
                print("Energy val :")
                print(z)
            print(abs(norm_Ax - mem), abs(rhok))
            if abs(norm_Ax - mem) < 100 and abs(rhok) < precision:
                print("Iteration converged:")
                print("    Energy: ", end='')
                print(abs(rhok))
                print("    matrix-vector norm: ", end='')
                print(norm_Ax)
                print("Eigenvalue found:")
                print(z)
                print("Eigenvector found:")
                print(x)
                ztab.append(z)
                xtab.append(x)
                ytab.append(y)
                converged = True
                break

            elif np.abs(z - z0) > maxrad or np.imag(z) < 0:
                print("Getting out of the research zone.")
                # Reset the energy to the initial energy
                bas_fun_index = bas_fun_index + 1
                z = z0
                if bas_fun_index >= len(x):
                    print("Functional space maximally expanded")
                else:
                    # Redefine the initial vector using a different basis function
                    # x = np.zeros(len(x0), dtype=complex)
                    x[bas_fun_index] = 1
                    x = x / lg.norm(x)
                    y = np.conj(x)

                if looping >= 3:
                    converged = False
                    "Looping through getting out the research zone"
                    break
                looping = looping + 2
            elif ite % maxit == 0 and ite != 0:
                print("Expanding functional space")
                bas_fun_index = bas_fun_index + 1
                if bas_fun_index >= len(x):
                    print("Functional space maximally expanded")
                else:
                    x[bas_fun_index] = 1
                    x = x / lg.norm(x)
                    y = np.conj(x)
                # z = z0
            elif np.isnan(z):
                raise ValueError("Undefined value of z. Terminating")

        if isinstance(zvec, list):
            num_eigval = num_eigval + 1
            if num_eigval + 1 >= len(zvec):
                convergence = False
                "Got all the expected eigenvalues. Break"
                break

        if not converged:
            print("The Nonlinear eigenvalue problem did not converge. ")
            print("Convergence at termination:")
            print("    Energy: ", end='')
            print(abs(rhok))
            print("    matrix-vector norm: ", end='')
            print(norm_Ax)
            print("Energy value at termination:")
            print(z)
            convergence = False

    return [ztab, xtab]


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


def pseudo_spectrum(A: callable):
    y = np.linspace(0, 0.25, 50)
    x = np.linspace(1.4, 2.1, 150)
    func_to_plot = partial(abs_det_norm, A)
    plot_functions.plot_2d_func(func_to_plot, x, y, part='real', scale='log')
    # plot_functions.plot_func(func_to_plot,x)


def inv_det_norm(A, x, y):
    num = lg.det(A(x + y * 1j))
    # if num < 1e-4:
    #    num = 1
    return 1 / (num + 1e-6)


def abs_det_norm(A, x, y):
    num = np.abs(lg.det(A(x + y * 1j)))
    # if num < 1e-4:
    #    num = 1
    return num
