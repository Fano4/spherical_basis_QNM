import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np


def plot_func(func_to_plot, x):
    f = np.zeros(x.shape, dtype=complex)
    for i in range(len(x)):
        f[i] = func_to_plot(x[i])
    plt.plot(x, np.real(f))
    plt.plot(x, np.imag(f))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    return 0


def plot_2d_func(func_to_plot, x, y, **kwargs):
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    ReZ = np.zeros(X.shape)
    ImZ = np.zeros(X.shape)

    for i in range(len(x)):
        for j in range(len(y)):
            out = func_to_plot(x[i], y[j])
            if isinstance(out, list):
                ReZ[j, i] = np.real(out[0])
                ImZ[j, i] = np.imag(out[0])
            else:
                ReZ[j, i] = np.real(out)
                ImZ[j, i] = np.imag(out)

    Z = ReZ + 1j * ImZ
    levels = MaxNLocator(nbins=15).tick_values(-abs(Z).max(), abs(Z).max())

    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    cmap = plt.get_cmap('RdBu')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig, (ax0, ax1) = plt.subplots(nrows=2)

    re = ax0.pcolormesh(X, Y, np.real(Z), cmap=cmap, norm=norm, shading='nearest')
    fig.colorbar(re, ax=ax0)
    ax0.set_title('Real part')

    # contours are *point* based plots, so convert our bound into point
    # centers
    im = ax1.pcolormesh(X, Y, np.imag(Z), cmap=cmap, norm=norm, shading='nearest')
    fig.colorbar(im, ax=ax1)
    ax1.set_title('Imaginary part')

    # adjust spacing between subplots so `ax1` title and `ax0` tick labels
    # don't overlap
    fig.tight_layout()

    plt.show()
    return 0
