"""Compare bSSFP and GRE contrasts."""

import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan

from ssfp import bssfp, spoiled_gre


if __name__ == '__main__':
    N = 128
    M0, T1, T2 = shepp_logan((N, N, 1), MR=True, zlims=(-.25, -.25))
    M0, T1, T2 = M0[..., 0], T1[..., 0], T2[..., 0]

    TR = 6e-3
    TE = TR/2
    alpha = np.deg2rad(70)
    bssfp_res = np.abs(bssfp(T1, T2, TR, alpha=alpha, M0=M0))
    gre_res = spoiled_gre(T1, T2, TR, TE, alpha=alpha, M0=M0)

    nx, ny = 1, 3
    plt.subplot(nx, ny, 1)
    plt.imshow(bssfp_res)
    plt.title('bSSFP')

    plt.subplot(nx, ny, 2)
    plt.imshow(gre_res)
    plt.title('GRE')

    plt.subplot(nx, ny, 3)
    idx = np.logical_and(bssfp_res > 0, gre_res > 0)
    plt.plot(bssfp_res[idx]/gre_res[idx])
    plt.ylim([0, 1])
    plt.title('bSSFP/GRE')

    plt.show()
