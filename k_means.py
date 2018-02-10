import numpy as np


def do_k_means(K, im, pix_mu, tol=1e-3, max_iters=200):
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
    # Colors of the centroids.
    mu_colors = im[pix_mu]
    new_mu_colors = np.zeros_like(mu_colors)

    new_im = np.zeros_like(im)
    for iteration in range(max_iters):
        # Distances to the centroids.
        distances = np.zeros((K, np.shape(im)[0]))
        for k in range(K):
            distances[k] = np.linalg.norm(im - mu_colors[k], axis=1)

        # Centroid pertencence.
        nearest_centroids = np.argmin(distances, axis=0)

        # New centroids.
        for k in range(K):
            ix_k = np.nonzero(nearest_centroids == k)
            new_mu_colors[k] = np.mean(im[ix_k], axis=0)

        distance_new_centroids = np.linalg.norm(new_mu_colors - mu_colors)
        print(distance_new_centroids)
        tol_reached = True if distance_new_centroids < tol else False

        mu_colors = np.copy(new_mu_colors)
        if tol_reached:
            for k in range(K):
                ix_k = np.nonzero(nearest_centroids == k)
                new_im[ix_k] = new_mu_colors[k]
            break

    return new_im
