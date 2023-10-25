"""Python implementation of region growing phase correction."""

import logging
from typing import Tuple

import numpy as np
from tqdm import trange

from ssfp.utils import choose_cntr


def scatterangle(xi: np.ndarray, yi: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    xi, yi : array_like
        Vectors of data point coordinates.

    Returns
    -------
    ang : array_like
        Angle of best fit line through origin, in radians.

    Notes
    -----
    Function fits the best line through a 2D scatter of
    data, minimizing the squared closest distance from each data
    point to the line.
    """

    sxx = np.sum(xi.flatten() ** 2)
    syy = np.sum(yi.flatten() ** 2)
    sxy = np.sum((xi * yi).flatten())

    # The equation that minimizes the sum of squared distances
    # is 0.5*tan(2*theta) = sxy / (sxx-syy).
    #
    # There is some ambiguity, since the theta that MAXIMIZES
    # the sum of squared distances is also a solution.
    #
    # However, since atan2(y,x) wraps only every 2*pi, this means
    # that theta is uniquely determined within a range of pi.
    # The solution with atan2(-y,-x) would give the theta that
    # maximizes the sum of squared distances.

    return 0.5 * np.arctan2(2 * sxy, sxx - syy)


def rgphcorr3d(im: np.ndarray, cellsize: Tuple[int]=(4, 4, 4), use_ctr: bool=False, slice_axis: int=-1, ret_extra: bool=False):
    """Region-growing phase correction for 3d complex image data.

    Parameters
    ----------
    im : array_like, 3-dimensional
        Array of complex pixel values from an SSFP acquisition.
    cellsize : list_like, optional
        Size of cubic region cells.
    use_ctr : bool, optional
        Automatically choose the center point as start.
        If use_ctr=False, use GUI to choose starting point in fat.
    slice_axis : int, optional
        Axis holding slices.  Only used if use_ctr=False.
    ret_extra : bool, optional
        Also return angles and weights.

    Returns
    -------
    pcim : array_like
        Image following phase correction.
    cellangle : array_like, optional
        Angle removed from each cell.
    cellweight : array_like, optional
        Weights for each cell (unused in current implementation).

    Notes
    -----
    We assume that the phase component of im consists
    of a slowly-spatially-varying component added to
    a random sign flip.  Beginning with a central, high-signal
    pixel, regions are added, and the slowly-varying phase
    is removed on a cell-by-cell basis.

    Implements the algorithm described in [1]_.  Based on the MATLAB
    implementation found at [2]_.

    References
    ----------
    .. [1] Hargreaves, Brian A., et al. "Fat‐suppressed steady‐state
           free precession imaging using phase detection." Magnetic
           Resonance in Medicine: An Official Journal of the
           International Society for Magnetic Resonance in Medicine
           50.1 (2003): 210-213.
    .. [2] http://mrsrl.stanford.edu/~brian/psssfp/
    """

    assert im.ndim == 3, 'im should be a 3-dimensional array!'
    assert len(cellsize) == 3, (
        'cellsize should be a tuple with 3 entries!')
    logging.debug('cellsize is %s', cellsize)
    cellsize = np.array(cellsize)

    # ======= Find Starting Cell by Displaying Image. ===========
    szim = np.array(im.shape[:])

    # Select pixel in fat
    if use_ctr:
        # Just start in image center.
        cntr = np.round(szim / 2)
    else:
        cntr = choose_cntr(np.abs(im), slice_axis=slice_axis)

    # Central cell, in cell coordinates.
    ccell = np.floor(cntr / cellsize + 1)

    # Duplicate array, for offset calcs.
    ccellbig = np.ones((27, 1)) * ccell[None, :]

    # ======= Make padded array with integer # of cells =======

    ncells = np.ceil(szim / cellsize).astype(int)
    for k in range(3):  # If 2x2x2, neighbour checks fail.
        if ncells[k] < 3:
            ncells[k] = 3

    # Pad the image if needed
    did_pad = False
    if not np.allclose(ncells * cellsize, szim):
        maxvox = ncells * cellsize
        logging.info(
            'Padding the image to <%d, %d, %d>',
            maxvox[0], maxvox[1], maxvox[2])

        # Do the actual padding
        px, py, pz = [x - y for x, y in zip(maxvox, im.shape)]
        adjx, adjy, adjz = np.mod([px, py, pz], 2)
        px2, py2, pz2 = int(px / 2), int(py / 2), int(pz / 2)
        im = np.pad(
            im,
            ((px2 + adjx, px2), (py2 + adjy, py2), (pz2 + adjz, pz2)),
            mode='constant')
        did_pad = True

    # ======= Output Variables =======

    # Allocate phase-corrected image.
    pcim = -.0001 * np.ones(im.shape, dtype=im.dtype)

    # Angles ultimately removed in correction:
    cellangle = np.zeros(ncells)

    # # Weights for each cell, calculated as sum of mags of dot
    # # products with angle.
    cellweight = np.zeros(ncells)

    # ======= Define coordinate arrays =======
    #
    # These are just arrays in the cell coordinates containing
    # the x, y and z locations of each block. When cells are reordered
    # based on distance from center, these maps will be used to get
    # the cell coordinates where the calculated angle is to be placed.

    # locations of cells, in cell coords
    xmap = np.zeros(ncells, dtype=int)
    ymap = np.zeros(ncells, dtype=int)
    zmap = np.zeros(ncells, dtype=int)

    # X locations.
    # m1 = [1:ncells(1)]'*ones(1,ncells(2));
    m1 = np.arange(1, ncells[0] + 1)[:, None] @ np.ones((1, ncells[1]))
    for k in range(ncells[2]):
        xmap[..., k] = m1

    # Y locations.
    # m1 = ones(ncells(1),1)*[1:ncells(2)];
    m1 = np.ones((ncells[0], 1)) @ np.arange(1, ncells[1] + 1)[None, :]
    for k in range(ncells[2]):
        ymap[..., k] = m1

    # Z locations.
    # m1 = ones(ncells(2),1)*[1:ncells(3)];
    m1 = np.ones((ncells[1], 1)) @ np.arange(1, ncells[2] + 1)[None, :]
    for k in range(ncells[0]):
        zmap[k, ...] = m1

    logging.debug(
        'xmap max/min - %d , %d',
        np.min(xmap.flatten()),
        np.max(xmap.flatten()))
    logging.debug(
        'ymap max/min - %d , %d',
        np.min(ymap.flatten()),
        np.max(ymap.flatten()))
    logging.debug(
        'zmap max/min - %d , %d',
        np.min(zmap.flatten()),
        np.max(zmap.flatten()))

    # ======= Figure out Distances of cells from center cell. =======
    #
    # Calculate distances of cells from center cell, and grow
    # region outward.  This guarantees that whenever the phase
    # in a cell is calculated, then the phases of all closer
    # cells have already been calculated.
    #

    # Make list of coordinates. Use Fortran ordering to match MATLAB
    # concatenation result
    cells = np.concatenate((
        xmap.flatten('F')[:, None],
        ymap.flatten('F')[:, None],
        zmap.flatten('F')[:, None]), axis=1)
    ccelle = np.ones((np.prod(ncells), 1)) @ ccell[None, :]

    # Delta, in cell coords, from center to cell.
    celldelta = cells - ccelle

    # Euclidean distance.
    celld = np.sqrt(np.sum(celldelta * celldelta, axis=-1))

    # Order distances from center, use mergesort to match MATLAB's
    # stable quicksort, see https://stackoverflow.com/questions/
    # 39484073/matlab-sort-vs-numpy-argsort-how-to-match-results
    cellorder = np.argsort(celld, kind='mergesort')

    # ======= Do phase fit and correction for cells in order. ========

    # Allocate "working cell"
    celldata = np.zeros(cellsize.shape, dtype=im.dtype)

    # Arrays of indices of starting corner for cell data in im.
    # cellstartX = ([1:ncells(1)]-1)*cellsize(1)
    # cellstartY = ([1:ncells(2)]-1)*cellsize(2)
    # cellstartZ = ([1:ncells(3)]-1)*cellsize(3)
    cellstartX = np.arange(ncells[0]) * cellsize[0]
    cellstartY = np.arange(ncells[1]) * cellsize[1]
    cellstartZ = np.arange(ncells[2]) * cellsize[2]

    # Offsets for cells. (+1 to match MATLAB indexing)
    celloffsetX = np.arange(cellsize[0]) + 1
    celloffsetY = np.arange(cellsize[1]) + 1
    celloffsetZ = np.arange(cellsize[2]) + 1

    # ====== Main Loop Initialization ========

    # Offsets to neighbours.
    nbr_offsts = np.array(
        [
            [1, 1, 1],
            [2, 1, 1],
            [3, 1, 1],
            [1, 2, 1],
            [2, 2, 1],
            [3, 2, 1],
            [1, 3, 1],
            [2, 3, 1],
            [3, 3, 1],
            [1, 1, 2],
            [2, 1, 2],
            [3, 1, 2],
            [1, 2, 2],
            [3, 2, 2],
            [1, 3, 2],
            [2, 3, 2],
            [3, 3, 2],
            [1, 1, 3],
            [2, 1, 3],
            [3, 1, 3],
            [1, 2, 3],
            [2, 2, 3],
            [3, 2, 3],
            [1, 3, 3],
            [2, 3, 3],
            [3, 3, 3],
        ])
    nbr_offsts = 2 * np.ones((26, 3)) - nbr_offsts
    nbr_dists = np.sqrt(np.sum(nbr_offsts * nbr_offsts, axis=-1))

    # ====== Main Loop =========

    # Do this for all cells.
    desc = 'Fitting cells'
    for k in trange(int(np.prod(ncells)), desc=desc, leave=False):

        cellnum = cellorder[k]
        cellnum_idx = np.unravel_index(cellnum, ncells, order='F')

        # ===== Extract Data Points from im.
        X, Y, Z = np.meshgrid(
            cellstartX[xmap[cellnum_idx] - 1] + celloffsetX - 1,
            cellstartY[ymap[cellnum_idx] - 1] + celloffsetY - 1,
            cellstartZ[zmap[cellnum_idx] - 1] + celloffsetZ - 1)
        celldata = im[X, Y, Z]

        # disp('CELL RANGE: ');
        # disp(cellstartX(xmap(cellnum))+celloffsetX);
        # disp(cellstartY(ymap(cellnum))+celloffsetY);
        # disp(cellstartZ(zmap(cellnum))+celloffsetZ);

        # Calculate angle of best fit line through points, and origin.
        # Angle is in the range [-pi/2, pi/2]
        an = scatterangle(celldata.real, celldata.imag)
        anv = np.exp(1j * an)

        # Transposition magic to match MATLAB flattening of 3-d array
        tmp = celldata.transpose((1, 0, 2)).flatten('F')
        tmp = np.concatenate((
            tmp.real[:, None],
            tmp.imag[:, None]), axis=1)
        magv = tmp @ np.array([anv.real, anv.imag])
        cellweight[cellnum_idx] = np.sum(np.abs(magv))

        # ===== Compare with weighted average of neighbors.

        # Dot products.
        ninds = nbr_offsts @ celldelta[cellnum, :]

        # Nec. for closer.
        f = ninds < 0
        cands = nbr_offsts[f, :]
        canddists = nbr_dists[f]
        ndelts = cands + (
                np.ones((np.sum(f), 1)) @ celldelta[cellnum, :][None, :])
        ndists = np.sqrt(np.sum(ndelts * ndelts, axis=-1))

        # f = find(ndists < celld(cellnum));
        f = ndists < celld[cellnum]
        ndelts = ndelts[f, :]  # Keep only closer cell-deltas.
        nseps = canddists[f]  # Sep from cell to neighbour.

        # cell coords of nbrs.
        ncoords = (ndelts + ccellbig[:np.sum(f), :]).astype(int)

        # Figure out the "average" angle of these by summing
        # a vector for each neighbour, whose length is reduced
        # by the distance separating the neighbour from the cell
        # of interest.
        vtot = 0
        if np.sum(f) > 0:
            logging.debug(
                'Cell at <%2d,%2d,%2d> ',
                cells[cellnum, 0],
                cells[cellnum, 1],
                cells[cellnum, 2])

            for p in range(np.sum(f)):
                logging.debug(
                    'Nearer Neighbour at <%2d,%2d,%2d> (sep = %f) ',
                    ncoords[p, 0],
                    ncoords[p, 1],
                    ncoords[p, 2],
                    nseps[p])

                if (all(ncoords[p, :] > 0) and
                        all(ncoords[p, :] <= ncells)):
                    vtot += 1 / nseps[p] * np.exp(1j * cellangle[
                        ncoords[p, 0] - 1,
                        ncoords[p, 1] - 1,
                        ncoords[p, 2] - 1])

            # "Average" neighbour angle.
            nang = np.arctan2(vtot.imag, vtot.real)

        else:
            # At Center point -- just take angle of average of scatter
            avgvec = np.mean(celldata.flatten())
            nang = np.arctan2(avgvec.imag, avgvec.real)

        # ===== Compare with phase of neighbors, and make sure
        # ultimate phase differs by less than pi/2.  The hope
        # here is to be insensitive to phase fluctuations of
        # +/- pi in the image, but only detect the slowly-varying
        # phase.

        # Angle difference.
        diff = an - nang

        # Modulus that to +/-(pi/2)
        rd = np.mod(diff + np.pi / 2, np.pi) - np.pi / 2

        # Keep |difference| < pi/2.
        corran = nang + rd

        # Keep ultimate angle in [-pi,pi]
        corran = np.mod(corran + np.pi, 2 * np.pi) - np.pi

        logging.debug(
            'Cell <%3d,%3d,%3d>, fit=%3d deg.  neighbor=%3d deg.  '
            'final=%3d deg.',
            xmap[cellnum_idx],
            ymap[cellnum_idx],
            zmap[cellnum_idx],
            np.round(180 / np.pi * an),
            np.round(180 / np.pi * nang),
            np.round(180 / np.pi * corran))

        # ===== Store and Correct phase
        cellangle[cellnum_idx] = corran

        # Correction phase factor.
        pf = np.exp(-1j * corran)
        pcim[X, Y, Z] = pf * celldata

    # End of main loop.

    # Remove padding if we added it:
    if did_pad:
        nx, ny, nz = pcim.shape[:]
        pcim = pcim[px2 + adjx:nx - px2, py2 + adjy:ny - py2, pz2 + adjz:nz - pz2]
        cellangle = cellangle[
                    px2 + adjx:nx - px2, py2 + adjy:ny - py2, pz2 + adjz:nz - pz2]

    if ret_extra:
        return pcim, cellangle, cellweight
    return pcim
