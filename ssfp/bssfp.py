"""Balanced SSFP simulation."""

from collections import defaultdict
from typing import Union, List, Tuple, Dict

import numpy as np


flt_or_array_like = Union[float, np.ndarray, List, Tuple]


def _result_shape(arg_shapes: List[Tuple[int, ...]]) -> Tuple[Tuple[int, ...], Dict[Tuple[int, ...], int]]:
    if not arg_shapes:
        return tuple(), dict()

    unique_shapes = []
    for sh in arg_shapes:
        if sh not in unique_shapes:
            unique_shapes.append(sh)

    result_dict = dict()
    result_sh = []
    idx = 0
    for sh in unique_shapes:
        result_sh += list(sh)
        result_dict[sh] = idx
        idx += len(sh)

    return tuple(result_sh), result_dict


def _isarray(arr):
    """Helper function to determine if arr is a numpy array."""
    return isinstance(arr, np.ndarray)


def _get_theta(TR, field_map, phase_cyc, delta_cs):
    """Get theta, spin phase per repetition time, given off-resonance.

    Parameters
    ----------
    TR : array_like
        repetition time (in sec).
    field_map : array_like
        Off-resonance map (in Hz).
    phase_cyc : array_like, optional
        Phase-cycling (in rad).
    delta_cs : float, optional
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
    """
    return 2*np.pi*(delta_cs + field_map)*TR + phase_cyc


def _get_bssfp_phase(T2, TR, field_map, delta_cs, phi_rf, phi_edd, phi_drift):
    """Additional bSSFP phase factors.

    Parameters
    ----------
    T2 : array_like
        Longitudinal relaxation constant (in sec).
    TR : array_like
        Repetition time (in sec).
    field_map : array_like
        off-resonance map (Hz).
    delta_cs : float, optional
        chemical shift of species w.r.t. the water peak (Hz).
    phi_rf : array_like, optional
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
    equation [5] of [3]_.

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
    """
    TE = TR/2  # assume bSSFP
    phi = 2*np.pi*(delta_cs + field_map)*TE + phi_rf + phi_edd + phi_drift

    # divide-by-zero risk
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.exp(1j*phi)*np.exp(-1*np.nan_to_num(TE/T2))


def bssfp(T1: flt_or_array_like, T2: flt_or_array_like, TR: flt_or_array_like, alpha: flt_or_array_like,
          field_map: flt_or_array_like=0, phase_cyc: flt_or_array_like=0, M0: flt_or_array_like=1, delta_cs: float=0,
          phi_rf: flt_or_array_like=0, phi_edd: float=0, phi_drift: float=0):
    r"""bSSFP transverse signal at time TE after excitation.

    Parameters
    ----------
    T1 : float or array_like
        longitudinal exponential decay time constant (in seconds).
    T2 : float or array_like
        transverse exponential decay time constant (in seconds).
    TR : float or array_like
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
    phi_rf : float or array_like, optional
        RF phase offset, related to the combin. of Tx/Rx phases (in
        rad).
    phi_edd : float, optional
        phase errors due to eddy current effects (in rad).
    phi_drift : float, optional
        phase errors due to B0 drift (in rad).

    Returns
    -------
    Mxy : numpy.ndarray
        Transverse complex magnetization.

    Notes
    -----
    `T1`, `T2`, `TR`, `alpha`, `field_map`, `phase_cyc`, `M0`, and `phi_rf` can all be
    either scalars or arrays.

    Output shape is determined by the shapes of input arrays.  All input
    arrays with equal shape will be assumed to have overlapping axes.  All
    input arrays with unique shapes will be assumed to have distinct axes
    and will be broadcast appropriately.

    Implementation of equations [1--2] in [1]_.  These equations are
    based on the Ernst-Anderson derivation [4]_ where off-resonance
    is assumed to be subtracted as opposed to added (as in the
    Freeman-Hill derivation [5]_).  Hoff actually gets Mx and My
    flipped in the paper, so we fix that here.  We also assume that
    the field map will be provided given the Freeman-Hill convention.

    We will additionally assume that linear phase increments
    (phase_cyc) will be given in the form:

    .. math::

        \theta = 2 \pi (\delta_{cs} + \Delta f_0)\text{TR} + \Delta \theta.

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
    """

    def _looks_iterable_but_is_not_ndarray(thing) -> bool:
        return not _isarray(thing) and (isinstance(thing, list) or isinstance(thing, tuple))

    # If it looks like an array, treat it like an array:
    if _looks_iterable_but_is_not_ndarray(T1):
        T1 = np.asarray(T1)
    if _looks_iterable_but_is_not_ndarray(T2):
        T2 = np.asarray(T2)
    if _looks_iterable_but_is_not_ndarray(TR):
        TR = np.asarray(TR)
    if _looks_iterable_but_is_not_ndarray(alpha):
        alpha = np.asarray(alpha)
    if _looks_iterable_but_is_not_ndarray(field_map):
        field_map = np.asarray(field_map)
    if _looks_iterable_but_is_not_ndarray(phase_cyc):
        phase_cyc = np.asarray(phase_cyc)
    if _looks_iterable_but_is_not_ndarray(M0):
        M0 = np.asarray(M0)
    if _looks_iterable_but_is_not_ndarray(phi_rf):
        phi_rf = np.asarray(phi_rf)

    # nominal case:
    #     all array_like arguments arrive with explicit, matching shapes
    #     no extra processing is needed, we just need to see that everything matches, and we're done
    # all different:
    #     all array_like arguments arrive with different shapes
    #     probably implies cross-product between all of them
    # some same, some different:
    #     assume shapes that align do indeed align; shapes are concatenated in the order the arguments arrive
    #     e.g., T1: (2, 3), T2: (2, 3), and phase_cyc: (3,) implies that the result will have shape (2, 3, 3)

    shapes = [arg.shape for arg in (T1, T2, TR, alpha, field_map, phase_cyc, M0, phi_rf) if _isarray(arg)]
    result_sh, result_sh_dict = _result_shape(arg_shapes=shapes)
    if _isarray(T1) and T1.shape != result_sh:
        idx = result_sh_dict[T1.shape]
        T1 = np.expand_dims(T1, list(range(idx)) + list(range(idx+T1.ndim, len(result_sh))))
    if _isarray(T2) and T2.shape != result_sh:
        idx = result_sh_dict[T2.shape]
        T2 = np.expand_dims(T2, list(range(idx)) + list(range(idx+T2.ndim, len(result_sh))))
    if _isarray(TR) and TR.shape != result_sh:
        idx = result_sh_dict[TR.shape]
        TR = np.expand_dims(TR, list(range(idx)) + list(range(idx+TR.ndim, len(result_sh))))
    if _isarray(alpha) and alpha.shape != result_sh:
        idx = result_sh_dict[alpha.shape]
        alpha = np.expand_dims(alpha, list(range(idx)) + list(range(idx+alpha.ndim, len(result_sh))))
    if _isarray(field_map) and field_map.shape != result_sh:
        idx = result_sh_dict[field_map.shape]
        field_map = np.expand_dims(field_map, list(range(idx)) + list(range(idx+field_map.ndim, len(result_sh))))
    if _isarray(phase_cyc) and phase_cyc.shape != result_sh:
        idx = result_sh_dict[phase_cyc.shape]
        phase_cyc = np.expand_dims(phase_cyc, list(range(idx)) + list(range(idx+phase_cyc.ndim, len(result_sh))))
    if _isarray(M0) and M0.shape != result_sh:
        idx = result_sh_dict[M0.shape]
        M0 = np.expand_dims(M0, list(range(idx)) + list(range(idx+M0.ndim, len(result_sh))))
    if _isarray(phi_rf) and phi_rf.shape != result_sh:
        idx = result_sh_dict[phi_rf.shape]
        phi_rf = np.expand_dims(phi_rf, list(range(idx)) + list(range(idx+phi_rf.ndim, len(result_sh))))

    # We are assuming Freeman-Hill convention for off-resonance map,
    # so we need to negate to make use with this Ernst-Anderson-
    # based implementation from Hoff
    field_map = -1 * field_map

    # We also assume that linear phase cycles will be added, but the
    # formulation used by Hoff, PLANET assumes subtracted, so let's
    # flip the signs
    phase_cyc = -1 * phase_cyc

    # divide-by-zero risk
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        E1 = np.exp(-1 * np.nan_to_num(TR / T1))
        E2 = np.exp(-1 * np.nan_to_num(TR / T2))

    # Precompute theta and some cos, sin
    theta = _get_theta(TR=TR, field_map=field_map, phase_cyc=phase_cyc, delta_cs=delta_cs)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    ct = np.cos(theta)
    st = np.sin(theta)

    # Get to business
    den = (1 - E1 * ca) * (1 - E2 * ct) - (E2 * (E1 - ca)) * (E2 - ct)
    Mx = -1 * M0 * ((1 - E1) * E2 * sa * st) / den
    My = M0 * ((1 - E1) * sa) * (1 - E2 * ct) / den
    Mxy = Mx + 1j * My

    # Add additional phase factor for readout at TE = TR/2.
    # Notice that phi_i are negated;
    # axes will be return in the order they are given
    return Mxy * _get_bssfp_phase(T2, TR, field_map, delta_cs, -1 * phi_rf, -1 * phi_edd, -1 * phi_drift)

    # # from https://arxiv.org/pdf/2302.12548.pdf:
    # # NOTE: below doesn't include phase factor at readout
    # a = M0*(1 - E1)*sa
    # b = 1 - E1*E2*E2 + (E2*E2 - E1)*ca
    # c = 2*(E1 - 1)*E2*np.cos(alpha/2)*np.cos(alpha/2)
    # field_map_rad = 2*np.pi*TR*field_map
    # # NOTE: switch phase_cyc sign from paper
    # return a/(b + c*np.cos(field_map_rad + phase_cyc))*(1 - E2*np.exp(-1j*(field_map_rad + phase_cyc)))
