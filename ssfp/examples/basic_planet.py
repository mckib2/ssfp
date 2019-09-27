'''Show basic usage of GS solution.'''

import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan
from tqdm import tqdm
from ssfp import bssfp, planet

if __name__ == '__main__':

    # Shepp-Logan
    N, npcs = 128, 8
    M0 = shepp_logan(N)
    T1, T2 = M0*2+1, M0/2

    # Simulate bSSFP acquisition with linear off-resonance
    TR, alpha = 3e-3, np.deg2rad(70)
    pcs = np.linspace(0, 2*np.pi, npcs, endpoint=False)
    df, _ = np.meshgrid(
        np.linspace(-1/TR, 1/TR, N),
        np.linspace(-1/TR, 1/TR, N))
    sig = bssfp(T1, T2, TR, alpha, field_map=df, phase_cyc=pcs, M0=M0)

    # Do T1, T2 mapping for each pixel
    mask = np.abs(M0.flatten()) > 1e-8
    idx = np.argwhere(mask).squeeze()
    sig = np.reshape(sig, (npcs, -1))
    T1map = np.zeros((N*N))
    T2map = np.zeros((N*N))
    for idx0 in tqdm(idx, leave=False, total=idx.size):
        _Meff, T1map[idx0], T2map[idx0] = planet(
            sig[:, idx0], alpha=alpha, TR=TR, T1_guess=1, pcs=pcs)
    mask = np.reshape(mask, (N, N))
    T1map = np.reshape(T1map, (N, N))
    T2map = np.reshape(T2map, (N, N))

    nx, ny = 2, 3
    plt.subplot(nx, ny, 1)
    plt.imshow(T1*mask)
    plt.title('T1 Truth')

    plt.subplot(nx, ny, 2)
    plt.imshow(T1map)
    plt.title('T1 est')

    plt.subplot(nx, ny, 3)
    plt.imshow(T1*mask - T1map)
    plt.title('Residual')

    plt.subplot(nx, ny, 4)
    plt.imshow(T2)
    plt.title('T2 Truth')

    plt.subplot(nx, ny, 5)
    plt.imshow(T2map)
    plt.title('T2 est')


    plt.subplot(nx, ny, 6)
    plt.imshow(T2 - T2map)

    plt.show()
