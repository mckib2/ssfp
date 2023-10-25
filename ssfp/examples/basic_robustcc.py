"""Show basic usage of Robust coil combination for bSSFP."""

from time import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from phantominator import shepp_logan

from ssfp import bssfp, robustcc, gs_recon


def _gaussian_csm(sx, sy, ncoil, sigma=1):
    """Simple coil model for demo."""
    X, Y = np.meshgrid(
        np.linspace(-1, 1, sx), np.linspace(-1, 1, sy))
    pos = np.stack((X[..., None], Y[..., None]), axis=-1)
    csm = np.empty((sx, sy, ncoil))
    cov = [[sigma, 0], [0, sigma]]
    for ii in range(ncoil):
        mu = [np.cos(ii/ncoil*np.pi*2), np.sin(ii/ncoil*2*np.pi)]
        csm[..., ii] = multivariate_normal(mu, cov).pdf(pos)
    return csm + 1j*csm


if __name__ == '__main__':
    # Coil combine params
    include_full_robustcc = True  # this might take a while to run

    # Sim params
    N, nc, npcs = 256, 8, 4
    pcs = np.linspace(0, 2*np.pi, npcs, endpoint=False)
    M0, T1, T2 = shepp_logan((N, N, 1), MR=True, zlims=(-.25, .25))
    M0, T1, T2 = np.squeeze(M0), np.squeeze(T1), np.squeeze(T2)

    # Linear off resonance
    TR, alpha = 6e-3, np.deg2rad(30)
    df, _ = np.meshgrid(
        np.linspace(-1/TR, 1/TR, N),
        np.linspace(-1/TR, 1/TR, N))

    # Generate coil images
    csm = _gaussian_csm(N, N, nc)
    data = np.abs(csm[..., None, :])*bssfp(
        T1, T2, TR, alpha, field_map=df, phase_cyc=pcs[None, None, :, None],
        M0=M0, phi_rf=np.angle(csm[..., None, :]))

    # Do coil-by-coil recon
    res_cbc = np.empty((N, N, nc), dtype=np.complex64)
    t0 = time()
    for ii in range(nc):
        res_cbc[..., ii] = gs_recon(data[..., ii], pc_axis=-1)
    res_cbc = np.sqrt(np.sum(np.abs(res_cbc)**2, axis=-1))
    print('Took %g seconds to do coil-by-coil recon' % (time() - t0))

    # Do robust coil combine then recon
    t0 = time()
    res_rcc_simple = robustcc(data, method='simple')
    res_rcc_simple = np.abs(gs_recon(res_rcc_simple, pc_axis=-1))
    print('Took %g seconds to do simple robustcc recon' % (time()-t0))

    if include_full_robustcc:
        t0 = time()
        res_rcc_full = robustcc(data, method='full', mask=M0 > 0)
        res_rcc_full = np.abs(gs_recon(res_rcc_full, pc_axis=-1))
        print(
            'Took %g seconds to do full robustcc recon' % (time()-t0))

    # Take a look
    nx, ny = 1, 2
    vmax = np.max(np.concatenate((res_cbc, res_rcc_simple)).flatten())
    if include_full_robustcc:
        vmax = np.maximum(vmax, np.max(res_rcc_full.flatten()))
        ny += 1
    plt_args = {
        'vmin': 0,
        'vmax': vmax,
        'cmap': 'gray',
    }
    plt.subplot(nx, ny, 1)
    plt.imshow(res_cbc, **plt_args)
    plt.title('Coil-by-coil')
    plt.axis('off')

    plt.subplot(nx, ny, 2)
    plt.imshow(res_rcc_simple, **plt_args)
    plt.title('Simple')
    plt.axis('off')

    if include_full_robustcc:
        plt.subplot(nx, ny, 3)
        plt.imshow(res_rcc_full, **plt_args)
        plt.title('Full')
        plt.axis('off')

    plt.show()
