'''Python implementation of region growing phase correction.'''

import logging

import numpy as np


def scatterangle(xi, yi):
    '''
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
    '''

    sxx = np.sum(xi.flatten()*xi.flatten())
    syy = np.sum(yi.flatten()*yi.flatten())
    sxy = np.sum(xi.flatten()*yi.flatten())

    #	The equation that minimizes the sum of squared distances
    #	is 0.5*tan(2*theta) = sxy / (sxx-syy).
    #
    #	There is some ambiguity, since the theta that MAXIMIZES
    #	the sum of squared distances is also a solution.
    #
    #	However, since atan2(y,x) wraps only every 2*pi, this means
    #	that theta is uniquely determined within a range of pi.
    #	The solution with atan2(-y,-x) would give the theta that
    #	maximizes the sum of squared distances.

    return 0.5*np.arctan2(2*sxy, sxx - syy)

def rgphcorr(im, cellsize):
    '''Region-growing phase correction to the given complex image data

    Parameters
    ----------
    im : array_like
        array of complex pixel values.
    cellsize : list
        Size of cubic region cells.

    Returns
    -------
    pcim : array_like
        Image following phase correction.
    cellangle : array_like
        Angle removed from each cell.

    Notes
    -----
    We assume that the phase component of im consists
    of a slowly-spatially-varying component added to
    a random sign flip.  Beginning with a central, high-signal
    pixel, regions are added, and the slowly-varying phase
    is removed on a cell-by-cell basis.

    References
    ----------
    '''

    assert len(cellsize) == 3, (
        'cellsize should be a list with 3 entries!')
    logging.debug('cellsize is %s', cellsize)
    cellsize = np.array(cellsize)

    # ======= Find Starting Cell by Displaying Image. ===========
    szim = np.array(im.shape[:])

    # TODO: select pixel in fat
    # print('Center Cursor in Fat (i.e. Bone) and right-click')
    # cntr, imlow, imhigh = disp3dmp(im) # Get center from disp3d program.
    #cntr = round(szim/2); % Just start in image center.
    cntr = np.round(szim/2)

    ccell = np.floor(cntr/cellsize) # Central cell, in cell coordinates.
    print(ccell)
    ccellbig = np.ones((27, 1))*ccell[None, :] # Duplicate array, for offset calcs.

    # ======= Make padded array with integer # of cells =======

    ncells = np.ceil(szim/cellsize).astype(int)
    print(ncells)
    for k in range(3): # If 2x2x2, neighbour checks fail.
        if ncells[k] < 3:
            ncells[k] = 3

    if np.allclose(ncells*cellsize, szim): # Pad the image.
        # a = 1 # Do nothing (Note that == checks all 3 dimensions).
        pass
    else:
        print('Padding the image to <%d, %d, %d>' % [c*cellsize for c in ncells])
        maxvox = ncells*cellsize
        im[maxvox[0], maxvox[1], maxvox[2]] = 0


    # ======= Output Variables =======

    pcim = -.0001*np.ones(im.shape, dtype=im.dtype) # Allocate phase-corrected image.

     # Angles ultimately removed in correction:
    cellangle = np.zeros(ncells)

    # Weights for each cell, calculated as sum of mags of dot
    # products with angle.
    cellweight = np.zeros(ncells)

    # ======= Define coordinate arrays =======
    #
    # These are just arrays in the cell coordinates containing
    # the x, y and z locations of each block. When cells are reordered
    # based on distance from center, these maps will be used to get
    # the cell coordinates where the calculated angle is to be placed.
    #
    xmap = np.zeros(ncells, dtype=int) # X-locations of cells, in cell coords
    ymap = np.zeros(ncells, dtype=int) # Y-locations of cells, in cell coords
    zmap = np.zeros(ncells, dtype=int) # Z-locations of cells, in cell coords

    # X locations.
    # m1 = [1:ncells(1)]'*ones(1,ncells(2));
    m1 = np.arange(ncells[0])[:, None] @ np.ones((1, ncells[1]))
    for k in range(ncells[2]):
        xmap[..., k] = m1

    # Y locations.
    # m1 = ones(ncells(1),1)*[1:ncells(2)];
    m1 = np.ones((ncells[0], 1)) @ np.arange(ncells[1])[None, :]
    for k in range(ncells[2]):
        ymap[..., k] = m1

   # Z locations.
    # m1 = ones(ncells(2),1)*[1:ncells(3)];
    m1 = np.ones((ncells[1], 1)) @ np.arange(ncells[2])[None, :]
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
    #	Calculate distances of cells from center cell, and grow
    #	region outward.  This guarantees that whenever the phase
    #	in a cell is calculated, then the phases of all closer
    #	cells have already been calculated.
    #

    # cells = [xmap(:) ymap(:) zmap(:)] # Make list of coordinates.
    cells = np.concatenate((
        xmap.flatten()[:, None],
        ymap.flatten()[:, None],
        zmap.flatten()[:, None]), axis=1)
    # ccelle = ones(prod(ncells),1)*ccell
    ccelle = np.ones((np.prod(ncells), 1)) @ ccell[None, :]
    celldelta = cells - ccelle # Delta, in cell coords, from center to cell.
    celld = np.sqrt(np.sum(celldelta*celldelta, axis=-1)) # Euclidean distance.

    # [orddists,cellorder] = sort(celld) # Order distances from center.
    # orddists = np.sort(celld)
    cellorder = np.argsort(celld)


    # ======= Do phase fit and correction for cells in order. ================

    celldata = np.zeros(cellsize.shape) # Allocate "working cell"

    # Arrays of indices of starting corner for cell data in im.
    # cellstartX = ([1:ncells(1)]-1)*cellsize(1)
    # cellstartY = ([1:ncells(2)]-1)*cellsize(2)
    # cellstartZ = ([1:ncells(3)]-1)*cellsize(3)
    cellstartX = np.arange(ncells[0])*cellsize[0]
    cellstartY = np.arange(ncells[1])*cellsize[1]
    cellstartZ = np.arange(ncells[2])*cellsize[2]

    celloffsetX = np.arange(cellsize[0]) # Offsets for cells.
    celloffsetY = np.arange(cellsize[1]) # Offsets for cells.
    celloffsetZ = np.arange(cellsize[2]) # Offsets for cells.

    # ====== Main Loop Initialization ========

    # Offsets to neighbours.
    nbr_offsts = np.array([
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
    nbr_offsts = 2*np.ones((26, 3)) - nbr_offsts
    nbr_dists = np.sqrt(np.sum(nbr_offsts*nbr_offsts, axis=-1))


    # ====== Main Loop =========

    for k in range(int(np.prod(ncells))): # Do this for all cells.
        logging.debug('---------------------------------------')

        if ((np.floor(100*(k+1)/np.prod(ncells)) - np.floor(100*k/np.prod(ncells)) > 0) or k == 0):
            pcdone = np.floor(100*k/np.prod(ncells))
            print('Fitting cells, %d%% Complete.' % pcdone)
    		# if ((display==1) and np.mod(pcdone,5) == 0):
    		# 	disp3dmp(pcim,imlow,imhigh,cntr,3);
    		# 	drawnow;
    		# 	# saveimage(pcdone);


        cellnum = cellorder.flatten()[k]

    	# ===== Extract Data Points from im.
        # print(xmap[])
        celldata = im[
            cellstartX[xmap[np.unravel_index(cellnum, xmap.shape)]] + celloffsetX,
            cellstartY[ymap[np.unravel_index(cellnum, ymap.shape)]] + celloffsetY,
            cellstartZ[zmap[np.unravel_index(cellnum, zmap.shape)]] + celloffsetZ]

    	#disp('CELL RANGE: ');
    	#disp(cellstartX(xmap(cellnum))+celloffsetX);
    	#disp(cellstartY(ymap(cellnum))+celloffsetY);
    	#disp(cellstartZ(zmap(cellnum))+celloffsetZ);

    	# ===== Calculate angle of best fit line through points, and origin.
    	# Angle is in the range [-pi/2, pi/2]
        an = scatterangle(
            np.real(celldata.flatten()),
            np.imag(celldata.flatten()))

        anv = np.exp(1j*an)
        # magv = [real(celldata(:)) imag(celldata(:))]*[real(anv);imag(anv)];
        magv = np.concatenate((
            np.real(celldata.flatten())[:, None],
            np.imag(celldata.flatten())[:, None]), axis=1) @ np.array([np.real(anv), np.imag(anv)])
        cellweight[np.unravel_index(cellnum, cellweight.shape)] = np.sum(np.abs(magv))

    	#plot(real(celldata(:)),imag(celldata(:)),'.');
    	#a = axis;
    	#axis(max(abs(a))*[-1 1 -1 1]);




    	# ===== Compare with weighted average of neighbors.

        ninds = nbr_offsts @ celldelta[cellnum, :].conj().T # Dot products.
    	# f = find(ninds < 0) # Nec. for closer.
        f = ninds < 0
        cands = nbr_offsts[f, :]
        canddists = nbr_dists[f]
        ndelts = cands + np.ones((np.sum(f), 1)) @ celldelta[cellnum, :][None, :]
        ndists = np.sqrt(np.sum(ndelts*ndelts, axis=-1))
    	# %if (debug > 1)
    		# %disp('Cell dist from center; neighbour distances.');
    		# %disp(celld(cellnum));
    		# %disp(ndists);
    	# %end;

    	# f = find(ndists < celld(cellnum));
        f = ndists < celld[np.unravel_index(cellnum, celld.shape)]
        ndelts = ndelts[f, :] # Keep only closer cell-deltas.
        nseps = canddists[f] # Sep from cell to neighbour.

        # ncoords = ndelts + ccellbig(1:length(f),:) # cell coords of nbrs.
        ncoords = ndelts + ccellbig[:np.sum(f), :]
        ncoords = ncoords.astype(int)

        vtot = 0

		# % Figure out the "average" angle of these by summing
		# % a vector for each neighbour, whose length is reduced
		# % by the distance separating the neighbour from the cell
		# % of interest.

        if np.sum(f) > 0:
            logging.debug('Cell at <%2d,%2d,%2d> ', cells[cellnum, :])
            for p in range(np.sum(f)):
                logging.debug('Nearer Neighbour at <%2d,%2d,%2d> (sep = %f) ', ncoords[p, :], nseps[p])
                if all(ncoords[p, :] > 0) and all(ncoords[p, :] <= ncells):
                    vtot += (1/nseps[p])*np.exp(1j*cellangle[ncoords[p, 0], ncoords[p, 1], ncoords[p, 2]])
    			  # %vtot = vtot + (1/nseps(p))*exp(i*cellangle(ncoords(p,1),ncoords(p,2),ncoords(p,3)))*cellweight(ncoords(p,1),ncoords(p,2),ncoords(p,3));
            nang = np.arctan2(np.imag(vtot), np.real(vtot)) # "Average" neighbour angle.

        else: # At Center point -- just take angle of average of scatter.
            avgvec = np.mean(celldata.flatten())
            nang = np.arctan2(np.imag(avgvec), np.real(avgvec))

    	# ===== Compare with phase of neighbors, and make sure
    	#	ultimate phase differs by less than pi/2.  The hope
    	#	here is to be insensitive to phase fluctuations of
    	#	+/- pi in the image, but only detect the slowly-varying
    	#	phase.

        diff = an - nang # Angle difference.
        rd = np.mod(diff + np.pi/2, np.pi) - np.pi/2 # Modulus that to +/-(pi/2)
        corran = nang + rd # Keep |difference| < pi/2.
        corran = np.mod(corran + np.pi, 2*np.pi) - np.pi # Keep ultimate angle in [-pi,pi]

        logging.debug(
            'Cell <%3d,%3d,%3d>, fit=%3d deg.  neighbor=%3d deg.  final=%3d deg.',
            xmap[np.unravel_index(cellorder[k], xmap.shape)],
            ymap[np.unravel_index(cellorder[k], ymap.shape)],
            zmap[np.unravel_index(cellorder[k], zmap.shape)],
            np.round(180/np.pi*an),
            np.round(180/np.pi*nang),
            np.round(180/np.pi*corran))

    	# ===== Store and Correct phase

        cellangle[np.unravel_index(cellnum, cellangle.shape)] = corran
        pf = np.exp(-1j*corran) # Correction phase factor.
        pcim[
            cellstartX[xmap[np.unravel_index(cellnum, xmap.shape)]] + celloffsetX,
            cellstartY[ymap[np.unravel_index(cellnum, ymap.shape)]] + celloffsetY,
            cellstartZ[zmap[np.unravel_index(cellnum, zmap.shape)]] + celloffsetZ] = pf*celldata


    # end;	% End of main loop.

    return(pcim, cellangle)
