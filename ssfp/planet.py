'''Python implementation of the PLANET algorithm.'''

import numpy as np

def planet(
        I, alpha, TR, T1_guess, pcs=None, fit_ellipse=None,
        compute_df=False, disp=False):
    '''Simultaneous T1, T2 mapping using phase‐cycled bSSFP.

    Parameters
    ----------
    I : array_like
        Complex voxels from phase-cycled bSSFP images.
    alpha : float
        Flip angle (in rad).
    TR : float
        Repetition time (in sec).
    T1_guess : float
        Estimate of expected T1 value (in sec).
    pcs : array_like, optional
        Phase-cycles that generate phase-cycle images of I (required
        if computing df) (in rad).
    fit_ellipse : callable, optional
        Function used to fit data points to ellipse.
    compute_df : bool, optional
        Whether or not estimate local off-resonance, df.
    disp : bool, optional
        Show debug plots.

    Returns
    -------
    Meff : array_like
        Effective magnetization amplitude (arbitrary units).
    T1 : array_like
        Estimate of T1 values (in sec).
    T2 : array_like
        Estimate of T2 values (in sec).
    df : array_like, optional
        Estimate of off-resonance values (in Hz).

    Raises
    ------
    AssertionError
        If fit_ellipse returns something that is not an ellipse
    AssertionError
        If the rotation fails and xc < 0 or yc =/= 0.
    AssertionError
        If a, b, or Meff are outside of interval (0, 1).
    ValueError
        If ellipse callapses to a line.
    ValueError
        If the sign of b cannot be determined.

    Notes
    -----
    Requires at least 6 phase cycles to fit the ellipse.  The ellipse
    fitting method they use (and which is implemented here) may not
    be the best method, but it is quick.  Could add more options for
    fitting in the future.

    fit_ellipse(x, y) should take two arguments and return a vector
    containing the coefficients of the implicit ellipse equation.  If
    fit_ellipse=None then the fit_ellipse_halir() function will be
    used.

    pcs should be a list of phase-cycles in radians.  If pcs=None, it
    will be determined as I.size equally spaced phasce-cycles on the
    interval [0, 2pi).

    Implements algorithm described in [1]_.

    References
    ----------
    .. [1] Shcherbakova, Yulia, et al. "PLANET: an ellipse fitting
           approach for simultaneous T1 and T2 mapping using
           phase‐cycled balanced steady‐state free precession."
           Magnetic resonance in medicine 79.2 (2018): 711-722.
    '''

    # Make sure we have an ellipse fitting function
    if fit_ellipse is None:
        fit_ellipse = _fit_ellipse_halir

    # Make sure we know what phase-cycles we have if we're computing
    # df
    if compute_df:
        if pcs is None:
            pcs = np.linspace(0, 2*np.pi, I.size, endpoint=False)
        else:
            # Make sure we get phase-cycles as a numpy array
            pcs = np.array(pcs)
        assert pcs.size == I.size, ('Number of phase-cycles must '
                                    'match entries of I!')

    ## Step 1. Direct linear least squares ellipse fitting to
    ## phase-cycled bSSFP data.
    C = fit_ellipse(I.real, I.imag)

    # Look at it in standard form
    C1, C2, C3, _C4, _C5, _C6 = C[:]
    assert C2**2 - 4*C1*C3 < 0, 'Not an ellipse!'

    ## Step 2. Rotation of the ellipse to initial vertical conic form.
    xr, yr, Cr, _phi = _do_planet_rotation(I)
    I0 = xr + 1j*yr
    xc, yc = _get_center(Cr)

    # Look at it to make sure we've rotated correctly
    if disp:
        import matplotlib.pyplot as plt
        Idraw = np.concatenate((I, [I[0]]))
        I0draw = np.concatenate((I0, [I0[0]]))
        plt.plot(Idraw.real, Idraw.imag, label='Sampled')
        plt.plot(I0draw.real, I0draw.imag, label='Rotated')
        plt.legend()
        plt.axis('square')
        plt.show()

    # Sanity check: make sure we got what we wanted:
    assert np.allclose(yc, 0), 'Ellipse rotation failed! yc = %g' % yc
    assert xc > 0, ('xc needs to be in the right half-plane! xc = %g'
                    '' % xc)
    # C1r, C2r, C3r, C4r, C5r, C6r = Cr[:]


    ## Step 3. Analytical solution for parameters Meff, T1, T2.
    # Get the semi axes, AA and BB
    A, B = _get_semiaxes(Cr)
    # Ellipse must be vertical -- so make the axes look like it
    if A > B:
        A, B = B, A
    A2 = A**2
    B2 = B**2

    # Decide sign of first term of b
    E1 = np.exp(-TR/T1_guess)
    aE1 = np.arccos(E1)
    if alpha > aE1:
        val = -1
    elif alpha < aE1:
        val = 1
    elif alpha == aE1:
        raise ValueError('Ellipse is a line! x = Meff')
    else:
        raise ValueError(
            'Houston, we should never have raised this error...')

    # See Appendix
    # xc = np.abs(xc) # THIS IS NOT IN THE APPENDIX but by def in
    # eq [9]
    xc2 = xc**2
    xcA = xc*A
    b = (val*xcA + np.sqrt(xcA**2 - (xc2 + B2)*(A2 - B2)))/(xc2 + B2)
    b2 = b**2
    a = B/(xc*np.sqrt(1 - b2) + b*B)
    ab = a*b
    Meff = xc*(1 - b2)/(1 - ab)

    # Sanity checks:
    assert 0 < b < 1, '0 < b < 1 has been violated! b = %g' % b
    assert 0 < a < 1, '0 < a < 1 has been violated! a = %g' % a
    assert 0 < Meff < 1, (
        '0 < Meff < 1 has been violated! Meff = %g' % Meff)

    # Now we can find the things we were really after
    ca = np.cos(alpha)
    T1 = -1*TR/(
        np.log((a*(1 + ca - ab*ca) - b)/(a*(1 + ca - ab) - b*ca)))
    T2 = -1*TR/np.log(a)

    ## Step 4. Estimation of the local off-resonance df.
    if compute_df:
        # The beta way:
        # costheta = np.zeros(dphis.size)
        # for nn in range(dphis.size):
        #     x, y = I0[nn].real, I0[nn].imag
        #     tanbeta = y/(x - xc)
        #     t = np.arctan(A/B*tanbeta)
        #     costheta[nn] = (np.cos(t) - b)/(b*np.cos(t) - 1)

        # The atan2 way:
        costheta = np.zeros(pcs.size)
        for nn in range(pcs.size):
            x, y = I0[nn].real, I0[nn].imag
            t = np.arctan2(y, x - xc)

            if a > b:
                costheta[nn] = (np.cos(t) - b)/(b*np.cos(t) - 1)
            else:
                # Sherbakova doesn't talk about this case in the
                # paper!
                costheta[nn] = (np.cos(t) + b)/(b*np.cos(t) + 1)

        # Get least squares estimate for K1, K2
        X = np.array([np.cos(pcs), np.sin(pcs)]).T
        K = np.linalg.multi_dot((np.linalg.pinv(
            X.T.dot(X)), X.T, costheta))
        K1, K2 = K[:]

        # And finally...
        theta0 = np.arctan2(K2, K1)
        df = -1*theta0/(2*np.pi*TR) # spurious negative sign, bug!
        return(Meff, T1, T2, df)

    # else...
    return(Meff, T1, T2)

def _get_semiaxes(c):
    '''Solve for semi-axes of the cartesian form of ellipse equation.

    Parameters
    ----------
    c : array_like
        Coefficients of general quadratic polynomial function for
        conic functions.

    Returns
    -------
    float
        Semi-major axis
    float
        Semi-minor axis

    Notes
    -----
    https://en.wikipedia.org/wiki/Ellipse
    '''
    A, B, C, D, E, F = c[:]
    B2 = B**2
    den = B2 - 4*A*C
    num = 2*(A*E**2 + C*D**2 - B*D*E + den*F)
    num *= (A + C + np.array([1, -1])*np.sqrt((A - C)**2 + B2))
    AB = -1*np.sqrt(num)/den

    # # Return semi-major axis first
    # if AB[0] > AB[1]:
        # print(AB)
        # return(AB[1], AB[0])
    return(AB[0], AB[1])

def _get_center(c):
    '''Compute center of ellipse from implicit function coefficients.

    Parameters
    ----------
    c : array_like
        Coefficients of general quadratic polynomial function for
        conic funs.

    Returns
    -------
    xc : float
        x coordinate of center.
    yc : float
        y coordinate of center.
    '''
    A, B, C, D, E, _F = c[:]
    den = B**2 - 4*A*C
    xc = (2*C*D - B*E)/den
    yc = (2*A*E - B*D)/den
    return(xc, yc)

def _rotate_points(x, y, phi, p=(0, 0)):
    '''Rotate points x, y through angle phi w.r.t. point p.

    Parameters
    ----------
    x : array_like
        x coordinates of points to be rotated.
    y : array_like
        y coordinates of points to be rotated.
    phi : float
        Angle in radians to rotate points.
    p : tuple, optional
        Point to rotate around.

    Returns
    -------
    xr : array_like
        x coordinates of rotated points.
    yr : array_like
        y coordinates of rotated points.
    '''
    x = x.flatten()
    y = y.flatten()
    xr = (x - p[0])*np.cos(phi) - (y - p[0])*np.sin(phi) + p[0]
    yr = (y - p[1])*np.cos(phi) + (x - p[1])*np.sin(phi) + p[1]
    return(xr, yr)

def _do_planet_rotation(I):
    '''Rotate complex pts to fit vertical ellipse centered at (xc, 0).

    Parameters
    ----------
    I : array_like
        Complex points from SSFP experiment.

    Returns
    -------
    xr : array_like
        x coordinates of rotated points.
    yr : array_like
        y coordinates of rotated points.
    cr : array_like
        Coefficients of rotated ellipse.
    phi : float
        Rotation angle in radians of effective rotation to get
        ellipse vertical and in the x > 0 half plane.
    '''

    # Represent complex number in 2d plane
    x = I.real.flatten()
    y = I.imag.flatten()

    # Fit ellipse and find initial guess at what rotation will make it
    # vertical with center at (xc, 0).  The arctan term rotates the
    # ellipse to be horizontal, then we need to decide whether to add
    # +/- 90 degrees to get it vertical.  We want xc to be positive,
    # so we must choose the rotation to get it vertical.
    c = _fit_ellipse_halir(x, y)
    phi = -.5*np.arctan2(c[1], (c[0] - c[2])) + np.pi/2
    xr, yr = _rotate_points(x, y, phi)

    # If xc is negative, then we chose the wrong rotation! Do -90 deg
    cr = _fit_ellipse_halir(xr, yr)
    if _get_center(cr)[0] < 0:
        # print('X IS NEGATIVE!')
        phi = -.5*np.arctan2(c[1], (c[0] - c[2])) - np.pi/2
        xr, yr = _rotate_points(x, y, phi)

    # Fit the rotated ellipse and bring yc to 0
    cr = _fit_ellipse_halir(xr, yr)
    yr -= _get_center(cr)[1]
    cr = _fit_ellipse_halir(xr, yr)
    # print(_get_center(cr))

    # With noisy measurements, sometimes the fit is incorrect in the
    # above steps and the ellipse ends up horizontal.  We can realize
    # this by finding the major and minor semiaxes of the ellipse.
    # The first axis returned should be the smaller if we were
    # correct, if not, do above steps again with an extra factor of
    # +/- 90 deg to get the ellipse standing up vertically.
    ax = _get_semiaxes(c)
    if ax[0] > ax[1]:
        # print('FLIPPITY FLOPPITY!')
        xr, yr = _rotate_points(x, y, phi + np.pi/2)
        cr = _fit_ellipse_halir(xr, yr)
        if _get_center(cr)[0] < 0:
            # print('X IS STILL NEGATIVE!')
            phi -= np.pi/2
            xr, yr = _rotate_points(x, y, phi)
        else:
            phi += np.pi/2

        cr = _fit_ellipse_halir(xr, yr)
        yr -= _get_center(cr)[1]
        cr = _fit_ellipse_halir(xr, yr)
        # print(_get_center(cr))

    return(xr, yr, cr, phi)

def _fit_ellipse_halir(x, y):
    '''Improved ellipse fitting algorithm by Halir and Flusser.

    Parameters
    ----------
    x : array_like
        y coordinates assumed to be on ellipse.
    y : array_like
        y coordinates assumed to be on ellipse.

    Returns
    -------
    array_like
        Ellipse coefficients.

    Notes
    -----
    Note that there should be at least 6 pairs of (x,y).

    From the paper's conclusion:

        "Due to its systematic bias, the proposed fitting algorithm
        cannot be used directly in applications where excellent
        accuracy of the fitting is required. But even in that
        applications our method can be useful as a fast and robust
        estimator of a good initial solution of the fitting
        problem..."

    See figure 2 from [2]_.
    '''

    # We should just have a bunch of points, so we can shape it into
    # a column vector since shape doesn't matter
    x = x.flatten()
    y = y.flatten()

    # Make sure we have at least 6 points (6 unknowns...)
    if x.size < 6 and y.size < 6:
        print((
            'WARNING: We need at least 6 sample points for a good '
            'fit!'))

    # Here's the heavy lifting
    D1 = np.stack((x**2, x*y, y**2)).T # quadratic pt of design matrix
    D2 = np.stack((x, y, np.ones(x.size))).T # lin part design matrix
    S1 = np.dot(D1.T, D1) # quadratic part of the scatter matrix
    S2 = np.dot(D1.T, D2) # combined part of the scatter matrix
    S3 = np.dot(D2.T, D2) # linear part of the scatter matrix
    T = -1*np.linalg.inv(S3).dot(S2.T) # for getting a2 from a1
    M = S1 + S2.dot(T) # reduced scatter matrix
    M = np.array([M[2, :]/2, -1*M[1, :], M[0, :]/2]) #premult by C1^-1
    _eval, evec = np.linalg.eig(M) # solve eigensystem
    cond = 4*evec[0, :]*evec[2, :] - evec[1, :]**2 # evaluate a’Ca
    a1 = evec[:, cond > 0] # eigenvector for min. pos. eigenvalue
    a = np.vstack([a1, T.dot(a1)]).squeeze() # ellipse coefficients
    return a

if __name__ == '__main__':
    pass
