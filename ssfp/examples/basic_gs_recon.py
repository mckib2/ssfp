"""Show basic usage of GS solution."""

import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan
from ssfp import bssfp, gs_recon


if __name__ == '__main__':
    # Shepp-Logan
    N = 128
    M0 = shepp_logan(N)
    T1, T2 = M0*2, M0/2

    # Simulate bSSFP acquisition with linear off-resonance
    TR, alpha = 3e-3, np.deg2rad(30)
    pcs = np.linspace(0, 2*np.pi, 4, endpoint=False)
    df, _ = np.meshgrid(
        np.linspace(-1/TR, 1/TR, N),
        np.linspace(-1/TR, 1/TR, N))
    sig = bssfp(T1, T2, TR, alpha, field_map=df,
                phase_cyc=pcs[None, None, :], M0=M0)

    # Show the phase-cycled images
    nx, ny = 2, 2
    plt.figure()
    for ii in range(nx*ny):
        plt.subplot(nx, ny, ii+1)
        plt.imshow(np.abs(sig[..., ii]))
        plt.title('%d deg PC' % (ii*90))
        plt.tick_params(axis='both', labelsize=0, length=0)
    plt.show(block=False)

    # Dhow the recon
    recon = gs_recon(sig, pc_axis=-1)
    plt.figure()
    plt.imshow(np.abs(recon))
    plt.title('GS Solution')
    plt.tick_params(axis='both', labelsize=0, length=0)
    plt.show()
