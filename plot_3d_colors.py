import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_3d_colors(im):
    # Select only a random subset of all pixels. All of them is very
    # computationally expensive.
    num_pix = 5000
    pix_mu = np.random.permutation(np.arange(np.shape(im)[0]))[:num_pix]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(im[pix_mu, 0], im[pix_mu, 1], im[pix_mu, 2], c=im[pix_mu]/256)
    plt.show()

    return
