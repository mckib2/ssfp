"""FIMTRE: FItting Multi-TR Ellipses."""

import numpy as np
from ssfp import gs_recon


def _fimtre6(I0, I1, TR0, TR1):
    ctr0 = gs_recon(I0, pc_axis=-1, second_pass=False)

    # find where circle intersects line, this is ctr1 (or there-abouts)
    # use equation of centered circle and line:
    #     x^2 + y^2 = r^2
    #     r = |GS(I0)|, where GS(I) is the geometric center of ellipse I
    #     (x0, y0) = 0 deg phase cycle of second TR ellipse
    #     (x1, y1) = 180 deg phase cycle of second TR ellipse
    #     y = mx + b
    #     m = (y1 - y0)/(x1 - x0)
    #     b = y - mx = y0 - m*x0 = y1 - m*x1
    #     x^2 + (mx + b)^2 = r^2
    #     => x =  (sqrt((m^2 + 1)r^2 - b^2) - bm)/(m^2 + 1)
    #         OR -(sqrt((m^2 + 1)r^2 - b^2) + bm)/(m^2 + 1)
    #        Choose the smaller rotation, i.e. min |x|
    #     y = mx + b
    #     (x, y) is now GS(I1)

    r2 = np.real(np.conj(ctr0)*ctr0)
    I1 = I1*1e8  # scale I1 up to make sure endpoints don't miss ctr0 when rotating
    m = (I1[..., 0].imag - I1[..., 1].imag)/(I1[..., 0].real - I1[..., 1].real)
    m2 = m*m
    # avg b = ((y_0 - mx_0) + (y_1 - mx_1))/2
    b = (I1[..., 0].imag - m*I1[..., 0].real + I1[..., 1].imag - m*I1[..., 1].real)/2
    vals = (m2 + 1)*r2 - b**2
    nonneg = vals >= 0
    x = np.zeros(r2.shape)
    xalt = np.zeros(r2.shape)
    x[nonneg] = (np.sqrt(vals[nonneg]) - b[nonneg]*m[nonneg])/(m2[nonneg] + 1)
    xalt[nonneg] = -1*(np.sqrt(vals[nonneg]) + b[nonneg]*m[nonneg])/(m2[nonneg] + 1)
    idx = np.abs(xalt) < np.abs(x)
    x[idx] = xalt[idx]
    y = m*x + b
    return np.angle(ctr0*(x - 1j*y))


def _fimtre8(I0, I1, TR0, TR1):
    ctr0 = gs_recon(I0, pc_axis=-1, second_pass=False)
    ctr1 = gs_recon(I1, pc_axis=-1, second_pass=False)
    return np.angle(ctr0*np.conj(ctr1))


def fimtre(I0: np.ndarray, I1: np.ndarray, TR0: float, TR1: float, pc_axis: int=-1, rad: bool=False):
    """Off-resonance using multiple TR ellipses.
    Parameters
    ----------
    I0 : array_like
        Complex-valued phase-cycled pixels with phase-cycles
        corresponding to [0, 90, 180, 270] degrees and TR0.
    I1 : array_like
        Complex-valued phase-cycled pixels with phase-cycles
        corresponding to [0, 180] or [0, 90, 180, 270]
        degrees and TR1.
    TR0, TR1 : float
        TR values in seconds corresponding to I0 and I1.
    pc_axis : int, optional
        Axis holding phase-cycle data.
    rad : bool, optional
        `rad=True` returns off-resonance in radians instead
         while `rad=False` return Hz.

    Returns
    -------
    theta : array_like
        Off-resonance estimate (Hz).

    Notes
    -----
    Uses 6 or 8 phase-cycled images to estimate off-resonance.
    """
    assert np.all(TR0 < TR1)
    I0 = np.moveaxis(np.atleast_2d(I0), pc_axis, -1)
    I1 = np.moveaxis(np.atleast_2d(I1), pc_axis, -1)
    assert I0.shape[-1] == 4
    assert I1.shape[-1] in {2, 4}
    if I1.shape[-1] == 4:
        theta = _fimtre8(I0, I1, TR0, TR1)
    else:
        theta = _fimtre6(I0, I1, TR0, TR1)
    if rad:
        return theta
    return np.array(1/(TR1/TR0 - 1)*theta/(np.pi*TR0))
