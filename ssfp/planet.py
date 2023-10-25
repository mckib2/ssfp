"""Python implementation of PLANET algorithm.

Notes
-----
Halir ellipse performs much faster! Somewhat different results though
between fast guaranteed and Halir -- it would be interesting to do an
analysis of the difference between the two methods and truth.
"""

from typing import Union

import numpy as np
from ellipsinator import fast_guaranteed_ellipse_estimate as ellipse_fit

flt_or_array_like = Union[float, np.ndarray]


def planet(I: np.ndarray, alpha: flt_or_array_like, TR: float, T1_guess: float=None, pcs: np.ndarray=None,
           mask: np.ndarray=None, pc_axis: int=-1, ret_all: bool=False):
    """Simultaneous T1, T2 mapping using phase‐cycled bSSFP.

    Parameters
    ----------
    I : array_like
        Complex phase-cycled bSSFP images. The data may be
        arbitrarily dimensional. The phase-cycle axis (pc_axis)
        holds the phase-cycle data.
    alpha : float or array_like
        Flip angle (in rad).
    TR : float
        Repetition time (in sec).
    T1_guess : float, optional
        Estimate of expected T1 value (in sec). If None, 1 sec
        is used as the default.
    pcs : array_like, optional
        The RF phase-cycle increments used.  If `None`,
        evenly spaced phase-cycles are assumed on [0, 2pi).
    mask : array_like or None, optional
        Which pixels of I to reconstruct.  If None, the mask
        reconstructs each pixel that has nonzero PC data.
    pc_axis : int, optional
        The axis that holds the phase-cycle data.
    ret_all : bool, optional
        Return all matrices return by original PLANET.

    Returns
    -------
    Meff : array_like
        Effective magnetization amplitude (arbitrary units).
    T1 : array_like
        Estimate of T1 values (in sec).
    T2 : array_like
        Estimate of T2 values (in sec).
    df : array_like
        Estimate of off-resonance (in Hz).
    phi : array_like, optional
    Xc : array_like, optional
    Yc : array_like, optional
    A : array_like, optional
    B : array_like, optional

    Notes
    -----
    Requires at least 6 phase cycles to fit the ellipse.  The ellipse
    fitting method they use (and which is implemented here) may not
    be the best method, but it is quick.  Could add more options for
    fitting in the future.

    Implements algorithm described in [1]_.

    References
    ----------
    .. [1] Shcherbakova, Yulia, et al. "PLANET: an ellipse fitting
           approach for simultaneous T1 and T2 mapping using
           phase‐cycled balanced steady‐state free precession."
           Magnetic resonance in medicine 79.2 (2018): 711-722.
    """
    # Phase cycles to the back
    I = np.moveaxis(I, pc_axis, -1)
    npcs = I.shape[-1]
    sh = I.shape[:-1]

    if pcs is None:
        pcs = np.linspace(0, 2*np.pi, npcs, endpoint=False)
    else:
        assert len(pcs) == npcs, "pcs and data must match!"

    # alpha can either be a scalar or array
    # Choose an initial estimate for T1
    if T1_guess is None:
        T1_guess = 1  # 1sec arbitrarily

    # Fit all ellipses that are nonzero
    if mask is None:
        recon_idx = np.nonzero(
            np.sum(np.abs(I).reshape((-1, npcs)), axis=-1))[0]
    else:
        recon_idx = np.nonzero(mask.flatten())[0]

    I = np.take(I.reshape((-1, npcs)), recon_idx, axis=0)
    ellipse_coefs = ellipse_fit(I)

    # Filter out all the fitting failures
    failure_idx = np.where(np.sum(np.abs(ellipse_coefs), axis=-1) == 0)[0]
    recon_idx = np.setdiff1d(recon_idx, failure_idx)
    ellipse_coefs = np.delete(ellipse_coefs, failure_idx, axis=0)

    # Find ellipse rotation
    A, B, C, D, E, F = ellipse_coefs.T
    phi = 1/2 * np.arctan(B/(A - C))

    c = np.cos(phi)
    s = np.sin(phi)
    c2, s2 = c**2, s**2

    A1 = A*c2 + B*c*s + C*s2
    D1 = D*c + E*s
    E1 = E*c - D*s
    C1 = A*s2 - B*c*s + C*c2
    F11 = F - ((D1**2)/(4*A1) + (E1**2)/(4*C1))

    Xc = -D1/(2*A1)
    Yc = -E1/(2*C1)
    aa = np.sqrt(-F11/A1)
    bb = np.sqrt(-F11/C1)

    # If phi needs to change then we need to recompute
    aa_le_bb = aa <= bb
    idx0 = np.logical_and(aa_le_bb, Xc < 0)
    phi[idx0] -= np.pi*np.sign(phi[idx0])

    Yc_ge_0 = Yc >= 0
    idx1 = np.logical_and(~aa_le_bb, Yc_ge_0)
    phi[idx1] += np.pi/2

    idx2 = np.logical_and(~aa_le_bb, ~Yc_ge_0)
    phi[idx2] -= np.pi/2

    idx = np.logical_or(idx0, np.logical_or(idx1, idx2))
    c[idx] = np.cos(phi[idx])
    s[idx] = np.sin(phi[idx])
    c2[idx], s2[idx] = c[idx]**2, s[idx]**2
    A1[idx] = A[idx]*c2[idx] + B[idx]*c[idx]*s[idx] + C[idx]*s2[idx]
    D1[idx] = D[idx]*c[idx] + E[idx]*s[idx]
    E1[idx] = E[idx]*c[idx] - D[idx]*s[idx]
    C1[idx] = A[idx]*s2[idx] - B[idx]*c[idx]*s[idx] + C[idx]*c2[idx]
    F11[idx] = F[idx] - ((D1[idx]**2)/(4*A1[idx]) + (E1[idx]**2)/(4*C1[idx]))
    Xc[idx] = -D1[idx]/(2*A1[idx])
    Yc[idx] = -E1[idx]/(2*C1[idx])
    aa[idx] = np.sqrt(-F11[idx]/A1[idx])
    bb[idx] = np.sqrt(-F11[idx]/C1[idx])

    # Decide sign of first term of b
    if isinstance(alpha, np.ndarray):
        bsign = np.ones(alpha.shape)
        bsign[alpha > np.arccos(np.exp(-TR/T1_guess))] = -1
    else:
        if alpha > np.arccos(np.exp(-TR/T1_guess)):
            bsign = -1
        else:
            bsign = 1

    # Compute interesting values
    Xc2 = Xc*Xc
    bb2 = bb*bb
    b = (bsign*Xc*aa + bb*np.sqrt(Xc2 - aa*aa + bb2))/(Xc2 + bb2)
    b2 = b*b
    a = bb/(b*bb + Xc*np.sqrt(1 - b2))
    ab = a*b
    M = Xc*(1 - b2)/(1 - ab)

    Mmap = np.zeros(np.prod(sh))
    Mmap[recon_idx] = M

    T2 = np.zeros(np.prod(sh))
    T2[recon_idx] = -TR/np.log(a)

    T1 = np.zeros(np.prod(sh))
    calpha = np.cos(alpha)
    T1[recon_idx] = -TR/np.log(
        ((a*(1 + calpha - ab*calpha) - b)/(a*(1 + calpha - ab) - b*calpha)))

    # compute off-resonance estimates
    Xn = I.real*c[..., None] + I.imag*s[..., None]
    Yn = I.imag*c[..., None] - I.real*s[..., None]
    Coef = (a - b)/(a*np.sqrt(1 - bb2))  # Coef=A/B
    TanT = Coef[..., None]*(Yn - Yc[..., None])/(Xn - Xc[..., None])
    T = np.arctan(TanT)  # defined on [-pi/2, pi/2]
    # unwrapping of T(k) to [-pi,pi]
    idx0 = Xn < Xc[..., None]
    idx_greater = np.logical_and(idx0, (Yn - Yc[..., None]) >= 0)
    # T[idx_greater] = np.pi - T
    np.putmask(T, idx_greater, np.pi - T[idx_greater])
    idx_lesser = np.logical_and(idx0, (Yn - Yc[..., None]) < 0)
    # T[idx_lesser] = -1*np.pi + T
    np.putmask(T, idx_lesser, -1*np.pi + T[idx_lesser])
    CosT = np.cos(T)
    c = (CosT - b[..., None])/(b[..., None]*CosT - 1)
    A = np.vstack((np.cos(pcs), np.sin(pcs))).T
    x = np.linalg.lstsq(A, c.T, rcond=None)[0]
    df = np.zeros(T1.shape)
    df[recon_idx] = -1*np.arctan2(x[1, :], x[0, :])/(2*np.pi*TR)

    if ret_all:
        # pack result matrices
        Phimap = np.zeros(np.prod(sh))
        Phimap[recon_idx] = phi
        Xcmap = np.zeros(np.prod(sh))
        Xcmap[recon_idx] = Xc
        Ycmap = np.zeros(np.prod(sh))
        Ycmap[recon_idx] = Yc
        Amap = np.zeros(np.prod(sh))
        Amap[recon_idx] = a
        Bmap = np.zeros(np.prod(sh))
        Bmap[recon_idx] = b

        return (
            np.reshape(Mmap, sh),
            np.reshape(T1, sh),
            np.reshape(T2, sh),
            np.reshape(df, sh),
            Phimap,
            Xcmap,
            Ycmap,
            Amap,
            Bmap,
        )

    return (
        np.reshape(Mmap, sh),
        np.reshape(T1, sh),
        np.reshape(T2, sh),
        np.reshape(df, sh),
    )


if __name__ == '__main__':
    pass
