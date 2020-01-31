'''Block regional phase correction.'''

import numpy as np
from skimage.util import view_as_windows

def brphcorr(im, block_size=(4, 4)):
    '''Python implementation.'''

    # 1) Divide image into small blocks
    blocks = view_as_windows(im, block_size)

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from phantominator import shepp_logan
    from ssfp import bssfp
    from ssfp import rgphcorr3d

    # Acquisiton params
    TR = 3e-3
    alpha = np.deg2rad(30)

    # Simulate without any banding
    N = 256
    M0, T1, T2 = shepp_logan((N, N, 4), MR=True, zlims=(-.25, 0))
    im = np.sum(
        bssfp(T1, T2, TR, alpha, phase_cyc=[0, np.pi], M0=M0), axis=0)
    pcim = rgphcorr3d(im, cellsize=(4, 4, 4), use_ctr=False, slice_axis=-1)

    plt.imshow((pcim[..., 0].real < 0)*np.abs(im[..., 0]))
    plt.show()

    assert False

    # Simulate two single-slice bSSFP acquisitons
    M0, T1, T2 = shepp_logan((N, N, 1), MR=True, zlims=(-.25, -.25))
    M0, T1, T2 = (x.squeeze() for x in [M0, T1, T2])

    xx = np.linspace(-1/TR, 1/TR, N)
    _, df = np.meshgrid(xx, xx)
    im0, im180 = bssfp(
        T1, T2, TR, alpha, field_map=df, phase_cyc=[0, np.pi], M0=M0)

    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(im0))
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(im180))
    plt.show()
