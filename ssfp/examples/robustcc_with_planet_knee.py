"""Show post-processing of data combined using robustcc."""

from time import time

import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_li

from ssfp import robustcc, gs_recon, planet


if __name__ == '__main__':
    # Load data
    data = np.load('/home/nicholas/Documents/research/ellipse_test/17.npy')
    data = data.transpose((0, 1, 3, 2))
    data = np.fft.fftshift(np.fft.fft2(data, axes=(0, 1)), axes=(0, 1))
    pdx2 = data.shape[0] // 4
    data = data[pdx2:-pdx2, ...]
    sx, sy, npcs, nc = data.shape[:]
    TR, alpha = 6e-3, np.deg2rad(70)

    # MR params for bSSFP sim
    pcs = np.linspace(0, 2*np.pi, npcs, endpoint=False)

    # Start timer
    t0 = time()

    # Coil combine (SOS + simple phase)
    res_rcc_simple = robustcc(data, method='simple', coil_axis=-1, pc_axis=-2)

    # Make a mask
    gs = np.array([gs_recon(res_rcc_simple[..., ii::4], pc_axis=-1) for ii in range(npcs // 4)])
    gs = np.abs(np.mean(gs, axis=0))
    thresh = threshold_li(gs)
    mask = gs > thresh

    # PLANET
    _Meff, T1map, T2map, _df = planet(res_rcc_simple, alpha, TR, T1_guess=1, mask=mask)

    # Stop timer
    print('Recon took %g sec' % (time() - t0))

    # Take a look
    plt.subplot(1, 2, 1)
    plt.imshow(T1map, vmin=0, vmax=1)
    plt.title('Est T1')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(T2map, vmin=0, vmax=1)
    plt.title('Est T2')
    plt.axis('off')

    plt.show()
