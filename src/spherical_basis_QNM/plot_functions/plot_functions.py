import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
from tqdm import tqdm


def plot_func(func_to_plot, x):
    f = np.zeros(x.shape, dtype=complex)
    for i in tqdm(range(len(x))):
        f[i] = func_to_plot(x[i])
    plt.plot(x * 1.23984193, np.real(f), label="Real part")
    plt.plot(x * 1.23984193, np.imag(f), label="Imaginary part")
    plt.xlabel("x", )
    plt.ylabel("y")
    plt.legend()
    plt.show()
    return 0


def plot_2d_func(func_to_plot, x, y, **kwargs):
    X, Y = np.meshgrid(x, y)
    ReZ = np.zeros(X.shape)
    ImZ = np.zeros(X.shape)

    for i in tqdm(range(len(x))):
        for j in range(len(y)):
            out = func_to_plot(x[i], y[j])
            if isinstance(out, list):
                ReZ[j, i] = np.real(out[0])
                ImZ[j, i] = np.imag(out[0])
            else:
                ReZ[j, i] = np.real(out)
                ImZ[j, i] = np.imag(out)

    Z = ReZ + 1j * ImZ

    Xp = X.reshape(len(x) * len(y))
    Yp = Y.reshape(len(x) * len(y))
    Zp = Z.reshape(len(x) * len(y))

    sav = np.array([Xp, Yp, Zp]).T
    np.savetxt("pseudo_spectrum.txt", sav)

    # Define the overall plot parameters
    # Font parameters
    mpl.rcParams['font.family'] = 'Avenir'
    mpl.rcParams['font.size'] = 18
    # Edit axes parameters
    mpl.rcParams['axes.linewidth'] = 2
    # Tick properties
    mpl.rcParams['xtick.major.size'] = 10
    mpl.rcParams['xtick.major.width'] = 2
    mpl.rcParams['xtick.direction'] = 'out'
    mpl.rcParams['ytick.major.size'] = 10
    mpl.rcParams['ytick.major.width'] = 2
    mpl.rcParams['ytick.direction'] = 'out'

    # Create figure and add axis
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    # ax2 = fig.add_subplot(212)

    # Set up the plot variables
    if 'scale' in kwargs and kwargs['scale'] == 'log':
        img = ax.imshow(ReZ, extent=(np.amin(X), np.amax(X), np.amin(Y), np.amax(Y)),
                        cmap='hot', norm=LogNorm(vmin=1, vmax=1e10))
    else:
        img = ax.imshow(ReZ, extent=(np.amin(X), np.amax(X), np.amin(Y), np.amax(Y)), cmap='hot', vmin=0,
                        vmax=np.amax(ReZ), zorder=1)

    # img2 = ax2.contourf(ImZ, 100, cmap='hot', vmin=0, vmax=np.amax(ImZ), zorder=1)

    # Set up color bar
    cbar_ax = make_axes_locatable(ax).append_axes(position='right', size='5%', pad=0.1)
    fig.colorbar(mappable=img, cax=cbar_ax)
    # cbar_ax2 = make_axes_locatable(ax2).append_axes(position='right', size='5%', pad=0.1)
    # cbar2 = fig.colorbar(mappable=img2, cax=cbar_ax2)
    plt.show()
    return 0
