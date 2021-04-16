'''Show basic usage of FIMTRE.'''

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import normalized_root_mse
from phantominator import shepp_logan

from ssfp import bssfp, gs_recon, fimtre


if __name__ == '__main__':
    # Shepp-Logan
    N, nslices = 256, 1
    M0, T1, T2 = shepp_logan((N, N, nslices), MR=True, zlims=(-.25, 0))
    M0, T1, T2 = np.squeeze(M0), np.squeeze(T1), np.squeeze(T2)
    mask = np.abs(M0) > 1e-8

    # Simulate bSSFP acquisition with linear off-resonance
    TR0, TR1 = 3e-3, 6e-3
    alpha = np.deg2rad(80)
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
    sig = 1e-5
    I0 += sig*(np.random.normal(0, 1, I0.shape) +
               1j*np.random.normal(0, 1, I0.shape))
    I1 += sig*(np.random.normal(0, 1, I1.shape) +
               1j*np.random.normal(0, 1, I1.shape))


    # Do the thing
    theta = fimtre(I0, I1, TR0, TR1, rad=False)*mask

    # reverse polarity if it makes sense
    if normalized_root_mse(theta, df) > normalized_root_mse(-1*theta, df):
        theta *= -1
    
    vmn, vmx = np.min(df.flatten()), np.max(df.flatten())
    opts = {'vmin': vmn, 'vmax': vmx}
    nx, ny = 1, 3
    plt.subplot(nx, ny, 1)
    plt.imshow(df, **opts)

    plt.subplot(nx, ny, 2)
    plt.imshow(theta, **opts)

    plt.subplot(nx, ny, 3)
    plt.imshow(df - theta, **opts)
    
    plt.show()
