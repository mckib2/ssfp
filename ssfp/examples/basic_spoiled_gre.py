"""Show basic usage of spoiled GRE."""

import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan

from ssfp import spoiled_gre


if __name__ == '__main__':
    N = 128
    M0, T1, T2 = shepp_logan((N, N, 1), MR=True, zlims=(-.25, -.25))
    M0, T1, T2 = M0[..., 0], T1[..., 0], T2[..., 0]

    TR, TE = 0.035, 0.01
    alpha = np.deg2rad(70)
    res = spoiled_gre(T1, T2, TR, TE, alpha=alpha, M0=M0)

    plt.imshow(res)
    plt.show()
