'''Show basic usage of GS solution.'''

import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan
from ssfp import bssfp, planet

if __name__ == '__main__':

    # Shepp-Logan
    N, nslices, npcs = 128, 2, 8 # 2 slices just to show we can
    M0, T1, T2 = shepp_logan((N, N, nslices), MR=True, zlims=(-.25, 0))

    # Simulate bSSFP acquisition with linear off-resonance
    TR, alpha = 3e-3, np.deg2rad(15)
    pcs = np.linspace(0, 2*np.pi, npcs, endpoint=False)
    df, _ = np.meshgrid(
        np.linspace(-1/TR, 1/TR, N),
        np.linspace(-1/TR, 1/TR, N))
    sig = np.empty((npcs,) + T1.shape, dtype='complex')
    for sl in range(nslices):
        sig[..., sl] = bssfp(T1[..., sl], T2[..., sl], TR, alpha, field_map=df, phase_cyc=pcs, M0=M0[..., sl])

    # Do T1, T2 mapping for each pixel
    mask = np.abs(M0) > 1e-8

    # Make it noisy
    sig += 1e-6*(np.random.normal(0, 1, sig.shape) + 1j*np.random.normal(0, 1, sig.shape))*mask

    # Do the thing
    Mmap, T1est, T2est = planet(sig, alpha, TR, mask=mask, pc_axis=0)

    # Look at a single slice
    sl = 0
    T1est = T1est[..., sl]
    T2est = T2est[..., sl]
    T1 = T1[..., sl]
    T2 = T2[..., sl]
    mask = mask[..., sl]

    nx, ny = 2, 3
    plt.subplot(nx, ny, 1)
    plt.imshow(T1*mask)
    plt.title('T1 Truth')
    plt.axis('off')

    plt.subplot(nx, ny, 2)
    plt.imshow(T1est)
    plt.title('T1 est')
    plt.axis('off')

    plt.subplot(nx, ny, 3)
    plt.imshow(T1*mask - T1est)
    plt.title('Residual')
    plt.axis('off')

    plt.subplot(nx, ny, 4)
    plt.imshow(T2)
    plt.title('T2 Truth')
    plt.axis('off')

    plt.subplot(nx, ny, 5)
    plt.imshow(T2est)
    plt.title('T2 est')
    plt.axis('off')

    plt.subplot(nx, ny, 6)
    plt.imshow(T2 - T2est)
    plt.axis('off')

    plt.show()
