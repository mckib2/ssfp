"""Geometric solution to the elliptical signal model."""

from typing import Tuple, Union

import numpy as np
from skimage.util.shape import view_as_windows


def gs_recon(Is: np.ndarray, pc_axis: int=0, isophase: float=np.pi, second_pass: bool=True,
             patch_size: Tuple[int]=None):
    """Full 2D Geometric Solution following Xiang, Hoff's 2014 paper.

    Parameters
    ----------
    Is : array_like
        4 phase-cycled images: [I0, I1, I2, I3].  (I0, I2) and
        (I1, I3) are phase-cycle pairs.
    pc_axis : int, optional
        Phase-cycle dimension, default is the first dimension.
    isophase : float, optional
        Only neighbours with isophase max phase difference contribute.
    second_pass : bool, optional
        Compute the second pass solution, increasing SNR by sqrt(2).
    patch_size : tuple, optional
        Size of patches in pixels (x, y).

    Returns
    -------
    I : array_like
        Second pass GS solution to elliptical signal model.
    Id : array_like, optional
        If second_pass=False, GS solution to the elliptical signal
        model (without linear weighted solution).

    Notes
    -----
    `Is` is an array of 4 2D images: I0, I1, I2, and I3.  I0 and I2
    make up the first phase-cycle pair, that is, they are 180 degrees
    phase-cycled relative to each other.  I1 and I3 are also
    phase-cycle pairs and must be different phase-cycles than either
    I0 or I2.  Relative phase-cycles are assumed as follows:

        - I0: 0 deg
        - I1: 90 deg
        - I2: 180 deg
        - I3: 270 deg

    Implements algorithm shown in Fig 2 of [1]_.

    References
    ----------
    .. [1] Xiang, Qing‐San, and Michael N. Hoff. "Banding artifact
           removal for bSSFP imaging with an elliptical signal
           model." Magnetic resonance in medicine 71.3 (2014):
           927-933.
    """

    # Put the pc_axis first
    Is = np.moveaxis(Is, pc_axis, 0)

    # Get direct geometric solution for demodulated M for all pixels
    Id = np.atleast_1d(_get_complex_cross_point(Is))

    # Get maximum pixel magnitudes for all input images
    I_max_mag = np.atleast_1d(np.max(np.abs(Is), axis=0))

    # Compute complex sum
    CS = np.atleast_1d(np.mean(Is, axis=0))

    # For each pixel, if the magnitude is greater than the maximum
    # magnitude of all four input images, then replace the pixel with
    # the CS solution.  This step regularizes the direct solution and
    # effectively removes all singularities
    mask = np.abs(Id) > I_max_mag
    Id[mask] = CS[mask]

    # Bail early if we don't want the second pass solution
    if not second_pass:
        return Id

    # Find weighted sums of image pairs (I1,I3) and (I2,I4)
    Iw13 = _compute_Iw(
        Is[0, ...], Is[2, ...], Id, patch_size=patch_size,
        isophase=isophase)
    Iw24 = _compute_Iw(
        Is[1, ...], Is[3, ...], Id, patch_size=patch_size,
        isophase=isophase)

    # Final result is found by averaging the two linear solutions for
    # reduced noise
    return (Iw13 + Iw24)/2


def _get_complex_cross_point(Is):
    """Find intersection of two lines connecting diagonal pairs.

    Parameters
    ----------
    Is : array_like
        4 phase-cycled images: [I0, I1, I2, I3].

    Returns
    -------
    M : array_like
        Complex cross point.

    Notes
    -----
    We assume that Is has the phase-cycle dimenension along the first
    axis.

    (xi, yi) are the real and imaginary parts of complex valued
    pixels in four bSSFP images denoted Ii and acquired with phase
    cycling dtheta = (i-1)*pi/2 with 0 < i < 4.

    This is Equation [13] from [1]_.
    """

    x1, y1 = Is[0, ...].real, Is[0, ...].imag
    x2, y2 = Is[1, ...].real, Is[1, ...].imag
    x3, y3 = Is[2, ...].real, Is[2, ...].imag
    x4, y4 = Is[3, ...].real, Is[3, ...].imag

    den = (x1 - x3)*(y2 - y4) + (x2 - x4)*(y3 - y1)
    if (den == 0).any():
        # Make sure we're not dividing by zero
        den += np.finfo(float).eps

    M = ((x1*y3 - x3*y1)*(Is[1, ...] - Is[3, ...]) - (
        x2*y4 - x4*y2)*(Is[0, ...] - Is[2, ...]))/den
    return M


def _compute_Iw(I0, I1, Id, patch_size=None, mode='constant', isophase=np.pi,
                ret_weight=False):
    """Computes weighted sum of image pair (I0, I1).

    Parameters
    ----------
    I0 : array_like
        1st of pair of diagonal images (relative phase cycle of 0).
    I1 : array_like
        2nd of pair of diagonal images (relative phase cycle of 180
        deg).
    Id : array_like
        result of regularized direct solution.
    patch_size : tuple, optional
        size of patches in pixels (x, y).  Defaults to (5, 5).
    mode : {'constant', 'edge'}, optional
        mode of numpy.pad. Probably choose 'constant' or 'edge'.
    isophase : float, optional
        Only neighbours with isophase max phase difference contribute.
    ret_weight : bool, optional
        Return weight, w.

    Returns
    -------
    Iw : array_like
        The weighted sum of image pair (I0,I1), equation [14]
    w : array_like, optional
        If ret_weight=True, w is returned.

    Notes
    -----
    Image pair (I0,I1) are phase cycled bSSFP images that are
    different by 180 degrees.  Id is the image given by the direct
    method (Equation [13]) after regularization by the complex sum.
    This function solves for the weights by regional differential
    energy minimization.  The 'regional' part means that the image is
    split into patches of size patch_size with edge boundary
    conditions (pads with the edge values given by mode option).  The
    weighted sum of the image pair is returned.

    The isophase does not appear in the paper, but appears in Hoff's
    MATLAB code.  It appears that we only want to consider pixels in
    the patch that have similar tissue properties - in other words,
    have similar phase.  The default isophase is pi as in Hoff's
    implementation.

    This function implements Equations [14,18], or steps 4--5 from
    Fig. 2 in [1]_.
    """

    # Make sure we have a patch size
    if patch_size is None:
        patch_size = (5, 5)

    # Expressions for the numerator and denominator
    numerator = np.conj(I1 - Id)*(I1 - I0) + np.conj(I1 - I0)*(
        I1 - Id)
    den = np.conj(I0 - I1)*(I0 - I1)

    # We'll have trouble with a 1d input if we don't do this
    numerator = np.atleast_2d(numerator)
    den = np.atleast_2d(den)

    # Pad the image so we can generate patches where we need them
    edge_pad = [int(p/2) for p in patch_size]
    numerator = np.pad(numerator, pad_width=edge_pad, mode=mode)
    den = np.pad(den, pad_width=edge_pad, mode=mode)

    # Separate out into patches of size patch_size
    numerator_patches = view_as_windows(numerator, patch_size)
    den_patches = view_as_windows(den, patch_size)

    # Make sure the phase difference is below a certan bound to
    # include point in weights
    mask = _mask_isophase(numerator_patches, patch_size, isophase)
    numerator_patches *= mask
    den_patches *= mask

    numerator_weights = np.sum(numerator_patches, axis=(-2, -1))
    den_weights = np.sum(den_patches, axis=(-2, -1))

    # Equation [18]
    weights = numerator_weights/(2*den_weights + np.finfo(float).eps)

    # Find Iw, the weighted sum of image pair (I0,I1), equation [14]
    Iw = I0*weights + I1*(1 - weights)
    if ret_weight:
        return Iw, weights
    return Iw


def _mask_isophase(numerator_patches, patch_size, isophase):
    """Generate mask that chooses patch pixels that satisfy isophase.

    Parameters
    ----------
    numerator_patches : array_like
        Numerator patches from second pass solution.
    patch_size : tuple
        size of patches in pixels (x,y).
    isophase : float
        Only neighbours with isophase max phase difference contribute.

    Returns
    -------
        mask : array_like
            same size as numerator_patches, to be applied to
            numerator_patches and den_patches before summation.
    """

    # # Loop through each patch and zero out all the values not
    # mask = np.ones(num_patches.shape).astype(bool)
    # center_x,center_y = int(patch_size[0]/2),int(patch_size[1]/2)
    # for ii in range(mask.shape[0]):
    #     for jj in range(mask.shape[1]):
    #         mask[ii,jj,...] = np.abs(np.angle(
    #            num_patches[ii,jj,...])*np.conj(
    #                num_patches[ii,jj,center_x,center_y])) < isophase
    # # print(np.sum(mask == False))

    # Now try it without loops - it'll be faster...
    center_x, center_y = [int(p/2) for p in patch_size]
    ref_pixels = np.repeat(np.repeat(
        numerator_patches[:, :, center_x, center_y, None],
        patch_size[0], axis=-1)[..., None], patch_size[1], axis=-1)
    mask_mat = np.abs(np.angle(
        numerator_patches)*np.conj(ref_pixels)) < isophase
    # assert np.allclose(mask_mat,mask)

    return mask_mat


if __name__ == '__main__':
    pass
