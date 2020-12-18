'''Estimate off-resonance using two TR ellipses.'''

import numpy as np
from ssfp import gs_recon


def _ormtre6(I0: np.ndarray, I1: np.ndarray,
             TR0: float, TR1: float,
             mask: np.ndarray, rad: bool) -> np.ndarray:
    sh = I0.shape[:-1]
    I0 = np.reshape(I0[mask], (-1, I0.shape[-1]))
    I1 = np.reshape(I1[mask], (-1, I1.shape[-1]))
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

    r2 = np.abs(ctr0)**2
    m = (I1[:, 0].imag - I1[:, 1].imag)/(I1[:, 0].real - I1[:, 1].real)
    m2 = m**2
    # avg b = ((y_0 - mx_0) + (y_1 - mx_1))/2
    b = (I1[:, 0].imag - m*I1[:, 0].real + I1[:, 1].imag - m*I1[:, 1].real)/2
    vals = (m2 + 1)*r2 - b**2
    nonneg = vals >= 0
    x = np.zeros(r2.shape)
    xalt = np.zeros(r2.shape)
    x[nonneg] = (np.sqrt(vals[nonneg]) - b[nonneg]*m[nonneg])/(m2[nonneg] + 1)
    xalt[nonneg] = -1*(np.sqrt(vals[nonneg]) + b[nonneg]*m[nonneg])/(m2[nonneg] + 1) #
    idx = np.abs(xalt) < np.abs(x)
    x[idx] = xalt[idx]
    y = m*x + b
    theta = np.zeros(sh)
    theta[mask] = np.angle(ctr0*(x - 1j*y))
    if rad:
        return theta
    return 1/(TR1/TR0 - 1)*theta/(np.pi*TR0)


def _ormtre8(I0: np.ndarray, I1: np.ndarray,
             TR0: float, TR1: float,
             mask: np.ndarray, rad: bool) -> np.ndarray:
    sh = I0.shape[:-1]
    I0 = np.reshape(I0[mask], (-1, I0.shape[-1]))
    I1 = np.reshape(I1[mask], (-1, I1.shape[-1]))

    # Find geometric centers of both ellipses
    ctr0 = gs_recon(I0, pc_axis=-1, second_pass=False)
    ctr1 = gs_recon(I1, pc_axis=-1, second_pass=False)
    theta = np.zeros(sh)
    theta[mask] = np.angle(ctr0*np.conj(ctr1))
    if rad:
        return theta
    return np.array(1/(TR1/TR0 - 1)*theta/(np.pi*TR0))


def ormtre(I0: np.ndarray, I1: np.ndarray,
           TR0: float, TR1: float,
           mask=None, pc_axis=-1, rad=False) -> np.ndarray:
    '''Off-resonance using multiple TR ellipses.

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
    mask : array_like, optional
        Boolean mask indicating which pixels to estimate
        off-resonance for.
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
    '''
    I0 = np.moveaxis(np.atleast_2d(I0), pc_axis, -1)
    I1 = np.moveaxis(np.atleast_2d(I1), pc_axis, -1)
    # print(I0.shape, I1.shape)
    assert I0.shape[-1] == 4, 'I0 must have 4 phase-cycles!'
    assert I1.shape[-1] in {2, 4}, 'I1 must have 2 or 4 phase-cycles!'

    # If caller has no mask in mind, do everything
    if mask is None:
        mask = np.ones(I0.shape[:-1], dtype=bool)
    assert mask.shape == I0.shape[:-1], 'Mask does not fit I0!'
    assert mask.shape == I1.shape[:-1], 'Mask does not fit I1!'

    # Assume TR0 < TR1
    assert TR0 < TR1, 'TR0 must be less than TR1!'

    if I1.shape[-1] == 2:
        return _ormtre6(I0, I1, TR0, TR1, mask=mask, rad=rad)
    return _ormtre8(I0, I1, TR0, TR1, mask=mask, rad=rad) # TODO: add mask


if __name__ == '__main__':
    from ssfp import bssfp
    TR1 = 3e-3
    mult = 1.15  # TR2 = mult*TR1 -- harmonic TR
    T1, T2 = 2, 1
    alpha = np.deg2rad(100)  # high flip angle
    # df = 1/(2.23234345*TR1)  # off-resonance -- what we're trying to estimate
    df = 1/(.25*TR1)
    TR2 = mult*TR1
    M0 = 1
    pcs0 = np.linspace(0, 2*np.pi, 4, endpoint=False)
    pcs1 = np.linspace(0, 2*np.pi, 2, endpoint=False)
    sig1 = bssfp(
        T1, T2, TR1, alpha, df, phase_cyc=pcs0, M0=M0,
        delta_cs=0, phi_rf=0, phi_edd=0, phi_drift=0)
    sig2 = bssfp(
        T1, T2, TR2, alpha, df, phase_cyc=pcs1, M0=M0,
        delta_cs=0, phi_rf=0, phi_edd=0, phi_drift=0)

    theta = ormtre(sig1, sig2, [1], TR1, TR2)
    print(df, theta)
