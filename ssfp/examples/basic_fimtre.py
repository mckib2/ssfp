'''Show basic usage of FIMTRE.'''

import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase
from phantominator import shepp_logan

from ssfp import bssfp, gs_recon, fimtre


if __name__ == '__main__':
    # Shepp-Logan
    N, nslices = 128, 1
    M0, T1, T2 = shepp_logan((N, N, nslices), MR=True, zlims=(-.25, 0))
    M0, T1, T2 = np.squeeze(M0), np.squeeze(T1), np.squeeze(T2)
    mask = np.abs(M0) > 1e-8

    # Simulate bSSFP acquisition with linear off-resonance
    TR0, TR1 = 3e-3, 3.1e-3
    alpha = np.deg2rad(120)
    pcs4 = np.linspace(0, 2*np.pi, 4, endpoint=False)
    pcs2 = np.linspace(0, 2*np.pi, 2, endpoint=False)
    df, _ = np.meshgrid(
        np.linspace(-1/(2*TR1), 1/(2*TR1), N),
        np.linspace(-1/(2*TR1), 1/(2*TR1), N))
    df *= mask
    I0 = bssfp(T1, T2, TR0, alpha, field_map=df,
               phase_cyc=pcs4[None, None, :], M0=M0)
    I1 = bssfp(T1, T2, TR1, alpha, field_map=df,
               phase_cyc=pcs2[None, None, :], M0=M0)

    # Make it noisy
    np.random.seed(0)
    sig = 0#1e-5
    I0 += sig*(np.random.normal(0, 1, I0.shape) +
               1j*np.random.normal(0, 1, I0.shape))
    I1 += sig*(np.random.normal(0, 1, I1.shape) +
               1j*np.random.normal(0, 1, I1.shape))

    # TODO: figure out wrapping
    theta = fimtre(I0, I1, TR0, TR1, rad=True)*mask
    theta = np.nan_to_num(theta)
    #theta = unwrap_phase(theta)
    theta = np.unwrap(theta, axis=0)
    theta = 1/(TR1/TR0 - 1)*theta/(np.pi*TR0)

    vmn, vmx = np.min(df.flatten()), np.max(df.flatten())
    opts = {'vmin': vmn, 'vmax': vmx}
    nx, ny = 3, 1
    plt.subplot(nx, ny, 1)
    plt.imshow(df, **opts)

    plt.subplot(nx, ny, 2)
    plt.imshow(theta, **opts)

    plt.subplot(nx, ny, 3)
    plt.imshow(df - theta, **opts)
    
    plt.show()
