import numpy as np
import matplotlib.pyplot as plt

from read_image import read_image
from k_means import do_k_means
from plot_3d_colors import plot_3d_colors

if __name__ == '__main__':
    im_name = './images/sheep_harris_2.jpg'
    # im_name = './images/torver_walk.jpg'
    # im_name = './images/green_spikey_thing_by_gill_santos.jpg'
    # im_name = './images/mountain_stream.jpg'
    im = read_image(im_name, True)
    im_shape = np.shape(im)
    num_pixels = im_shape[0]*im_shape[1]
    # Flatten the pixel locations (but not the color dimensions).
    im = np.reshape(im, (num_pixels, im_shape[2]))
    # Normalize the RGB colors from 0-255 to 0-1.
    im = im/256

    # Number of clusters/colors.
    K = 8

    plot_3d_colors(im)

    # We do the K-means several times with different inizialitaions
    # and then we will choose the one with the lowest cost.
    max_iter = 2
    for i in range(max_iter):
        # Initial centroids, in pixel coordinates.
        pix_mu = np.random.permutation(np.arange(num_pixels))[:K]

        do_k_means(K, im, pix_mu)

    # Start several times with different inizialitaions.
    # Choose inizialitaion.
    # Run K-means algorithm.

    # Select best compression.

    # Reshape the image into its original form.
    im = np.reshape(im, im_shape)
    # Show image.
    plt.imshow(im)
    plt.show()
