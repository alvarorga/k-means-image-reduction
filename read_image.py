import imageio
import numpy as np
import matplotlib.pyplot as plt


def read_image(im_name, show_image=False):
    """Write the image 'im_name' into a NumPy array."""
    im = imageio.imread(im_name)
    im_shape = np.shape(im)
    print('Read image with {}x{} pixels.'
          .format(im_shape[0], im_shape[1]))

    if show_image:
        plt.imshow(im)
        plt.show()

    return im
