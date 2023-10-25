"""Robust coil combination for bSSFP."""

import numpy as np
from tqdm import tqdm


def robustcc(data: np.ndarray, method: str='simple', mask: np.ndarray=None, coil_axis: int=-1,
             pc_axis: int=-2) -> np.array:
    """Robust elliptical-model-preserving coil combination for bSSFP.

    Parameters
    ----------
    data : array_like
        Complex bSSFP image space data to be coil combined.
    method : str, {'simple', 'full'}, optional
        The method of phase estimation.  method='simple' is very fast
        and does a good job.
    mask : array_like or None, optional
        Only used for method='full'.  Gives mask of which pixels to
        do fit on.  Does not have coil, phase-cycle dimensions.
    coil_axis : int, optional
        Dimension holding the coil data.
    pc_axis : int, optional
        Dimension holding the phase-cycle data.

    Returns
    -------
    res : array_like
        Complex coil-combined data that preserves elliptical
        relationships between phase-cycle pixels.

    Notes
    -----
    Implements the method described in [1]_.  This coil combination
    method preserves elliptical relationships between phase-cycle
    pixels for more efficient computation, e.g., gs_recon which
    reconstructs one coil at a time ([2]_).

    References
    ----------
    .. [1] N. McKibben, G. Tarbox, E. DiBella, and N. K. Bangerter,
           "Robust Coil Combination for bSSFP Elliptical Signal
           Model," Proceedings of the 28th Annual Meeting of the
           ISMRM; Sydney, NSW, Australia, 2020.
    .. [2] Xiang, Qing‐San, and Michael N. Hoff. "Banding artifact
           removal for bSSFP imaging with an elliptical signal model."
           Magnetic resonance in medicine 71.3 (2014): 927-933.
    """

    # Put coil and phase-cycle axes where we expect them
    data = np.moveaxis(data, (coil_axis, pc_axis), (-1, -2))

    # Use SOS for magnitude estimate
    mag = np.sqrt(np.sum(np.abs(data)**2, axis=-1))

    # Choose the method for phase estimation
    if method == 'simple':
        # simple coil phase estimate is fast:
        phase = _simple_phase(data)
    elif method == 'full':
        # Use the full solution:
        phase = _full_phase(data, mask)
    else:
        raise NotImplementedError()

    return np.moveaxis(
        mag*np.exp(1j*phase), (-1, -2), (coil_axis, pc_axis))


def _simple_phase(data):
    """Simple strategy: choose phase from best coil ellipse."""

    # Assume the best coil is the one with max median value
    idx = np.argmax(
        np.median(np.abs(data), axis=-1, keepdims=True), axis=-1)
    return np.take_along_axis(
        np.angle(data), idx[..., None], axis=-1).squeeze()


def _full_phase(data, mask=None):
    """Do pixel-by-pixel ellipse registration."""

    # Worst case is to do all pixels:
    if mask is None:
        mask = np.ones(data.shape[:-2], dtype=bool)

    # Do pixel-by-pixel phase estimation
    phase = np.empty(data.shape[:-1])
    for idx in tqdm(
            np.argwhere(mask),
            total=np.prod(np.sum(mask.flatten())),
            leave=False):

        # Register all coil ellipses to a single reference
        coil_ellipses = data[tuple(idx) + (slice(None), slice(None))]

        # Take reference ellipse to be the one with the greatest median value
        ref_idx = np.argmax(np.median(np.abs(
            coil_ellipses), axis=-1, keepdims=True), axis=-1)
        ref = np.take_along_axis(
            coil_ellipses, ref_idx[..., None], axis=-1).squeeze()

        # Do coil by coil registration
        reg_ellipses = np.empty(
            coil_ellipses.shape, dtype=coil_ellipses.dtype)
        W = np.empty(coil_ellipses.shape[-1])  # weights
        for cc in range(coil_ellipses.shape[-1]):
            T = np.linalg.lstsq(
                coil_ellipses[..., cc][:, None],
                ref, rcond=None)[0]
            W[cc] = np.abs(T)**2  # save the weights
            T = np.exp(1j*np.angle(T))  # just rotate, no scaling
            reg_ellipses[..., cc] = T*coil_ellipses[..., cc]

        # Take the weighted average to the composite ellipse
        phase[tuple(idx) + (slice(None),)] = np.average(
            np.angle(reg_ellipses), axis=-1, weights=W)

    # make sure undefined values are set to 0; we have to do this
    # since we allocated with np.empty
    phase[~mask] = 0  # pylint: disable=E1130
    return phase


if __name__ == '__main__':
    pass
