'''bSSFP simulation.'''

import numpy as np


def bssfp(
        T1, T2, TR, alpha, field_map=0, phase_cyc=0, M0=1, delta_cs=0,
        phi_rf=0, phi_edd=0, phi_drift=0, target_pc_axis=0):
    r'''bSSFP transverse signal at time TE after excitation.

    Parameters
    ----------
    T1 : float or array_like
        longitudinal exponential decay time constant (in seconds).
    T2 : float or array_like
        transverse exponential decay time constant (in seconds).
    TR : float
        repetition time (in seconds).
    alpha : float or array_like
        flip angle (in rad).
    field_map : float or array_like, optional
        B0 field map (in Hz).
    phase_cyc : float or array_like, optional
        Linear phase-cycle increment (in rad).
    M0 : float or array_like, optional
        proton density.
    delta_cs : float, optional
        chemical shift of species w.r.t. the water peak (in Hz).
    phi_rf : float, optional
        RF phase offset, related to the combin. of Tx/Rx phases (in
        rad).
    phi_edd : float, optional
        phase errors due to eddy current effects (in rad).
    phi_drift : float, optional
        phase errors due to B0 drift (in rad).
    target_pc_axis : int, optional
        Where the new phase-cycle dimension should be inserted.  Only
        used if phase_cyc is an array.

    Returns
    -------
    Mxy : numpy.array
        Transverse complex magnetization.

    Notes
    -----
    `T1`, `T2`, `alpha`, `field_map`, and `M0` can all be either a
    scalar or an MxN array.  `phase_cyc` can be a scalar or length L
    vector.

    Implementation of equations [1--2] in [1]_.  These equations are
    based on the Ernst-Anderson derivation [4]_ where off-resonance
    is assumed to be subtracted as opposed to added (as in the
    Freeman-Hill derivation [5]_).  Hoff actually gets Mx and My
    flipped in the paper, so we fix that here.  We also assume that
    the field map will be provided given the Freeman-Hill convention.

    We will additionally assume that linear phase increments
    (phase_cyc) will be given in the form:

    .. math::

        \theta = 2 \pi (\delta_{cs} + \Delta f_0)\text{TR} + \Delta
        \theta.

    Notice that this is opposite of the convention used in PLANET,
    where phase_cyc is subtracted (see equation [12] in [2]_).

    Also see equations [2.7] and [2.10a--b] from [4]_ and equations
    [3] and [6--12] from [5]_.

    References
    ----------
    .. [1] Xiang, Qing‐San, and Michael N. Hoff. "Banding artifact
           removal for bSSFP imaging with an elliptical signal
           model." Magnetic resonance in medicine 71.3 (2014):
           927-933.

    .. [4] Ernst, Richard R., and Weston A. Anderson. "Application of
           Fourier transform spectroscopy to magnetic resonance."
           Review of Scientific Instruments 37.1 (1966): 93-102.

    .. [5] Freeman R, Hill H. Phase and intensity anomalies in
           fourier transform NMR. J Magn Reson 1971;4:366–383.
    '''

    # We are assuming Freeman-Hill convention for off-resonance map,
    # so we need to negate to make use with this Ernst-Anderson-
    # based implementation from Hoff
    field_map = -1*field_map

    # We also assume that linear phase cycles will be added, but the
    # formulation used by Hoff, PLANET assumes subtracted, so let's
    # flip the signs
    phase_cyc = -1*phase_cyc

    # Make sure we're working with arrays
    T1 = np.atleast_2d(T1)
    T2 = np.atleast_2d(T2)
    alpha = np.atleast_2d(alpha)
    field_map = np.atleast_2d(field_map)
    phase_cyc = np.atleast_2d(phase_cyc)

    # If we have more than one phase-cycle, then add that dimension
    if phase_cyc.size > 1:
        reps = (phase_cyc.size, 1, 1)
        phase_cyc = np.tile(
            phase_cyc, T1.shape[:] + (1,)).transpose((2, 0, 1))
        T1 = np.tile(T1, reps)
        T2 = np.tile(T2, reps)
        alpha = np.tile(alpha, reps)
        field_map = np.tile(field_map, reps)

    # All this nonsense so we don't divide by 0
    E1 = np.zeros(T1.shape)
    E1[T1 > 0] = np.exp(-TR/T1[T1 > 0])
    E2 = np.zeros(T2.shape)
    E2[T2 > 0] = np.exp(-TR/T2[T2 > 0])

    # Precompute theta and some cos, sin
    theta = _get_theta(TR, field_map, phase_cyc, delta_cs)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    ct = np.cos(theta)
    st = np.sin(theta)

    # Get to business
    den = (1 - E1*ca)*(1 - E2*ct) - (E2*(E1 - ca))*(E2 - ct)
    Mx = -M0*((1 - E1)*E2*sa*st)/den
    My = M0*((1 - E1)*sa)*(1 - E2*ct)/den
    Mxy = Mx + 1j*My

    # Add additional phase factor for readout at TE = TR/2.
    # Notice that phi_i are negated
    Mxy *= _get_bssfp_phase(
        T2, TR, field_map, delta_cs, -phi_rf, -phi_edd, -phi_drift)

    # If multiple phase-cycles are being generated, move them to
    # specified axis
    Mxy = Mxy.squeeze()
    if phase_cyc.size > 1:
        # phase-cycle dimension currently in 0th position
        Mxy = np.moveaxis(Mxy, 0, target_pc_axis)

    return Mxy


def _get_bssfp_phase(
        T2, TR, field_map, delta_cs=0, phi_rf=0, phi_edd=0,
        phi_drift=0):
    '''Additional bSSFP phase factors.

    Parameters
    ----------
    T2 : array_like
        Longitudinal relaxation constant (in sec).
    TR : float
        Repetition time (in sec).
    field_map : array_like
        off-resonance map (Hz).
    delta_cs : float, optional
        chemical shift of species w.r.t. the water peak (Hz).
    phi_rf : float, optional
        RF phase offset, related to the combin. of Tx/Rx phases (rad).
    phi_edd : float, optional
        phase errors due to eddy current effects (rad).
    phi_drift : float, optional
        phase errors due to B0 drift (rad).

    Returns
    -------
    phase : array_like
        Additional phase term to simulate readout at time TE = TR/2.
        Assumes balanced (TE = TR/2).

    Notes
    -----
    This is exp(-i phi) from end of p. 930 in [1]_.

    We use a positive exponent, exp(i phi), as in Hoff and Taylor
    MATLAB implementations.  This phase factor is also positive in
    equaiton [5] of [3]_.

    In Hoff's paper the equation is not explicitly given for phi, so
    we implement equation [5] that gives more detailed terms, found
    in [2]_.

    References
    ----------
    .. [2] Shcherbakova, Yulia, et al. "PLANET: An ellipse fitting
           approach for simultaneous T1 and T2 mapping using
           phase‐cycled balanced steady‐state free precession."
           Magnetic resonance in medicine 79.2 (2018): 711-722.

    .. [3] Scheffler, Klaus, and Jürgen Hennig. "Is TrueFISP a
           gradient‐echo or a spin‐echo sequence?." Magnetic
           Resonance in Medicine: An Official Journal of the
           International Society for Magnetic Resonance in Medicine
           49.2 (2003): 395-397.
    '''

    TE = TR/2  # assume bSSFP
    phi = 2*np.pi*(
        delta_cs + field_map)*TE + phi_rf + phi_edd + phi_drift

    T2 = np.array(T2)
    idx = np.where(T2 > 0)
    val = np.zeros(T2.shape)
    val[idx] = -TE/T2[idx]
    return np.exp(1j*phi)*np.exp(val)


def _get_theta(TR, field_map, phase_cyc=0, delta_cs=0):
    '''Get theta, spin phase per repetition time, given off-resonance.

    Parameters
    ----------
    TR : float
        repetition time (in sec).
    field_map : array_like
        Off-resonance map (in Hz).
    phase_cyc : array_like, optional
        Phase-cycling (in rad).
    delta_cs : float, optional, optional
        Chemical shift of species w.r.t. the water peak (Hz).

    Returns
    -------
    theta : array_like
        Spin phase per repetition time, given off-resonance.

    Notes
    -----
    Equation for theta=2*pi*df*TR is in Appendix A of [6]_.  The
    additional chemical shift term can be found, e.g., in [2]_.

    References
    ----------
    .. [6] Hargreaves, Brian A., et al. "Characterization and
           reduction of the transient response in steady‐state MR
           imaging." Magnetic Resonance in Medicine: An Official
           Journal of the International Society for Magnetic
           Resonance in Medicine 46.1 (2001): 149-158.
    '''

    return 2*np.pi*(delta_cs + field_map)*TR + phase_cyc


if __name__ == '__main__':
    pass
