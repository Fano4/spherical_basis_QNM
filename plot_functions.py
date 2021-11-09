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

    ReZ = np.zeros(X.shape)
    ImZ = np.zeros(X.shape)

    for i in range(len(x)):
        for j in range(len(y)):
            out = func_to_plot(x[i], y[j])
            if isinstance(out, list):
                ReZ[i, j] = np.real(out[0])
                ImZ[i, j] = np.imag(out[0])
            else:
                ReZ[i, j] = np.real(out)
                ImZ[i, j] = np.imag(out)

    if 'part' in kwargs:
        if kwargs['part'] == 'real':
            fig, ax = plt.subplots()
            im = ax.imshow(ReZ, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()],
                           vmax=ReZ.max(), vmin=-ReZ.max())
        elif kwargs['part'] == 'imag':
            fig, ax = plt.subplots()
            im = ax.imshow(ImZ, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()],
                           vmax=ImZ.max(), vmin=-ImZ.max())
    else:
        fig, ax = plt.subplots()
        Z = ReZ + 1j * ImZ
        im = ax.imshow(Z, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()],
                       vmax=abs(Z.max()), vmin=-abs(Z.max()))

    plt.show()
    return 0
