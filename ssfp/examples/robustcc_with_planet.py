"""Show post-processing of data combined using robustcc."""

from time import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from phantominator import shepp_logan

from ssfp import bssfp, robustcc, planet


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
    # Sim params
    N, ncoils, npcs = 256, 64, 8
    TR, alpha = 4e-3, np.deg2rad(120)

    # Create coil sensitivities
    mps = _gaussian_csm(N, N, ncoils)

    # MR params for bSSFP sim
    pcs = np.linspace(0, 2*np.pi, npcs, endpoint=False)
    M0, T1, T2 = shepp_logan((N, N, 1), MR=True, zlims=(-.25, .25))
    M0, T1, T2 = np.squeeze(M0), np.squeeze(T1), np.squeeze(T2)

    # Linear off resonance -- exaggerate off-resonance effects
    df, _ = np.meshgrid(
        np.linspace(-1/TR, 1/TR, N),
        np.linspace(-1/TR, 1/TR, N))

    # Simulate the bSSFP acquisition
    data = np.abs(mps[..., None, :])*bssfp(
        T1, T2, TR=TR, alpha=alpha, field_map=df,
        phase_cyc=pcs[None, None, :], M0=M0,
        phi_rf=np.angle(mps[..., None, :]))

    # Add noise
    sigma = 1e-7
    data += (np.random.normal(0, sigma, size=data.shape) +
             1j*np.random.normal(0, sigma, size=data.shape))

    # Start timer
    t0 = time()

    # Coil combine (SOS + simple phase)
    res_rcc_simple = robustcc(data, method='simple', coil_axis=-1, pc_axis=-2)

    # PLANET
    mask = np.abs(M0) > 1e-8
    _Meff, T1map, T2map, _df = planet(res_rcc_simple, alpha, TR, T1_guess=1, mask=mask, pc_axis=-1)

    # Stop timer
    print('Recon took %g sec' % (time() - t0))

    # Take a look
    plt.subplot(2, 3, 1)
    plt.imshow(T1)
    plt.title('True T1')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(T1map)
    plt.title('Est T1')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(100*(T1map - T1)/(T1 + 1e-10))
    plt.title('Percent diff')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(T2)
    plt.title('True T2')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(T2map)
    plt.title('Est T2')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(100*(T2map - T2)/(T2 + 1e-10))
    plt.title('Percent diff')
    plt.colorbar()
    plt.axis('off')

    plt.show()
