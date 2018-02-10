import numpy as np


def do_k_means(K, im, pix_mu, opts=None):
    """Do the K-means algorithm to group the pixels of an image in K
    clusters.

    Args:
        K (int): number of clusters.
        im (ndarray of floats): image to clusterize.
        pix_mu (1darray of ints): pixel positions of the K centroids.
        opts (dict): parameters for the convergence of the algorithm.

    Returns:
        new_im (ndarray of floats): new image with clusterized colors.
    """
    # Convergence parameters.
    tol = 1e-3
    max_iters = 200
    if isinstance(opts, dict):
        if 'tol' in opts:
            tol = opts['tol']
        if 'max_iters' in opts:
            max_iters = opts['max_iters']

    # Colors of the centroids.
    mu_colors = im[pix_mu]
    for iteration in range(max_iters):
        # Array with distances

        # Select the points of the clusters closest to the centroids

        # Compute new centroids

        # Compute distance to centroids and test if has converged
        pass


    return None
