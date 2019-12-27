'''Basic usage of region growing phase correction.'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from ssfp import rgphcorr

if __name__ == '__main__':

    # Load sample data
    data = loadmat('data/samplepsssfp.mat')['dat']
    print(data.shape)

    # plt.imshow(np.abs(data[50, ...]))
    # plt.show()

    pcim, cellangle = rgphcorr(data, [4, 4, 4])

    plt.imshow(np.abs(pcim))
    plt.show()

    plt.imshow(np.real(pcim))
    plt.show()
