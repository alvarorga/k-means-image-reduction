import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_3d_colors(im):
    # Select only a random subset of all pixels. All of them is very
    # computationally expensive.
    num_pix = 5000 if np.shape(im)[0] > 5000 else np.shape(im)[0]
    pix_mu = np.random.permutation(np.arange(np.shape(im)[0]))[:num_pix]

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(im[pix_mu, 0], im[pix_mu, 1], im[pix_mu, 2], c=im[pix_mu])

    U, s, Vh = svd(np.cov(im.T))
    print(s/np.sum(s))
    proj = U[:, :2]

    proj_im = np.dot(im, proj)

    plt.figure(2)
    plt.scatter(proj_im[pix_mu, 0], proj_im[pix_mu, 1], c=im[pix_mu])
    plt.show()

    return
