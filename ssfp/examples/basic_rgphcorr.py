"""Basic usage of region growing phase correction."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from ssfp import rgphcorr3d
from ssfp.utils import download_file


if __name__ == '__main__':

    # Load sample data
    filepath = download_file(
        'http://mrsrl.stanford.edu/~brian/psssfp/samplepsssfp.mat',
        'samplepsssfp.mat')
    data = loadmat(filepath)['dat']

    # Do the phase correction
    pcim = rgphcorr3d(data, cellsize=(9, 6, 6), slice_axis=0)

    # Choose a slice to look at
    sl = 100
    plt.subplot(1, 2, 1)
    plt.imshow(np.real(data[sl, ...]))
    plt.axis('off')
    plt.title('Real(Image)')

    plt.subplot(1, 2, 2)
    plt.imshow(np.real(pcim[sl, ...]))
    plt.axis('off')
    plt.title('Real(Phase Corrected Image)')
    plt.show()

    # Look at water and fat images
    water_mask = pcim[sl, ...].real >= 0
    fat_mask = pcim[sl, ...].real < 0
    plt.subplot(1, 2, 1)
    plt.imshow(water_mask*np.abs(data[sl, ...]))
    plt.axis('off')
    plt.title('Water Image')

    plt.subplot(1, 2, 2)
    plt.imshow(fat_mask*np.abs(data[sl, ...]))
    plt.axis('off')
    plt.title('Fat Image')
    plt.show()
