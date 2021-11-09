import matplotlib.pyplot as plt
import numpy as np


def plot_func(func_to_plot, x):
    plt.plot(x, np.real(func_to_plot(x)))
    plt.plot(x, np.imag(func_to_plot(x)))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    return 0


def plot_2d_func(func_to_plot, x, y, **kwargs):
    X, Y = np.meshgrid(x, y)
    Z = func_to_plot(X, Y)

    if 'part' in kwargs:
        if kwargs['part'] == 'real':
            fig, ax = plt.subplots()
            im = ax.imshow(np.real(Z), origin='lower', extent=[x.min(), x.max(), y.min(), y.max()],
                           vmax=np.real(Z.max()), vmin=-np.real(Z.max()))
        elif kwargs['part'] == 'imag':
            fig, ax = plt.subplots()
            im = ax.imshow(np.imag(Z), origin='lower', extent=[x.min(), x.max(), y.min(), y.max()],
                           vmax=np.imag(Z.max()), vmin=-np.imag(Z.max()))
    else:
        fig, ax = plt.subplots()
        im = ax.imshow(Z, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()],
                       vmax=abs(Z.max()), vmin=-abs(Z.max()))

    plt.show()
    return 0
