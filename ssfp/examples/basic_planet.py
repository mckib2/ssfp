"""Show basic usage of GS solution."""

from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import normalized_root_mse
from skimage.restoration import unwrap_phase
from phantominator import shepp_logan
from ssfp import bssfp, planet


if __name__ == '__main__':
    # Shepp-Logan
    N, nslices, npcs = 128, 2, 8  # 2 slices just to show we can
    M0, T1, T2 = shepp_logan((N, N, nslices), MR=True, zlims=(-.25, 0))

    # Simulate bSSFP acquisition with linear off-resonance
    TR, alpha = 3e-3, np.deg2rad(15)
    pcs = np.linspace(0, 2*np.pi, npcs, endpoint=False)
    df, _ = np.meshgrid(
        np.linspace(-1/TR, 1/TR, N),
        np.linspace(-1/TR, 1/TR, N))
    sig = bssfp(T1, T2, TR, alpha, field_map=df[..., None],
                phase_cyc=pcs[None, None, None, :], M0=M0)

    # Do T1, T2 mapping for each pixel
    mask = np.abs(M0) > 1e-8

    # Make it noisy
    np.random.seed(0)
    sig += 1e-5*(np.random.normal(0, 1, sig.shape) +
                 1j*np.random.normal(0, 1, sig.shape))*mask[..., None]

    # Do the thing
    t0 = perf_counter()
    Mmap, T1est, T2est, dfest = planet(sig, alpha, TR, mask=mask, pc_axis=-1)
    print('Took %g sec to run PLANET' % (perf_counter() - t0))

    # Look at a single slice
    sl = 0
    T1est = T1est[..., sl]
    T2est = T2est[..., sl]
    dfest = dfest[..., sl]
    T1 = T1[..., sl]
    T2 = T2[..., sl]
    mask = mask[..., sl]

    # Simple phase unwrapping of off-resonance estimate
    dfest = unwrap_phase(dfest*2*np.pi*TR)/(2*np.pi*TR)

    nx, ny = 3, 3
    opts = {'cmap': 'gray'}
    plt.subplot(nx, ny, 1)
    plt.imshow(T1*mask, **opts)
    plt.title('T1 Truth')
    plt.axis('off')

    plt.subplot(nx, ny, 2)
    plt.imshow(T1est, **opts)
    plt.title('T1 est')
    plt.axis('off')

    plt.subplot(nx, ny, 3)
    plt.imshow(T1*mask - T1est, **opts)
    plt.title('NRMSE: %g' % normalized_root_mse(T1, T1est))
    plt.axis('off')

    plt.subplot(nx, ny, 4)
    plt.imshow(T2*mask, **opts)
    plt.title('T2 Truth')
    plt.axis('off')

    plt.subplot(nx, ny, 5)
    plt.imshow(T2est, **opts)
    plt.title('T2 est')
    plt.axis('off')

    plt.subplot(nx, ny, 6)
    plt.imshow(T2*mask - T2est, **opts)
    plt.title('NRMSE: %g' % normalized_root_mse(T2, T2est))
    plt.axis('off')

    plt.subplot(nx, ny, 7)
    plt.imshow(df*mask, **opts)
    plt.title('df Truth')
    plt.axis('off')

    plt.subplot(nx, ny, 8)
    plt.imshow(dfest, **opts)
    plt.title('df est')
    plt.axis('off')

    plt.subplot(nx, ny, 9)
    plt.imshow(df*mask - dfest, **opts)
    plt.title('NRMSE: %g' % normalized_root_mse(df*mask, dfest))
    plt.axis('off')

    plt.show()
