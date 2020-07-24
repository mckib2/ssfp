
import numpy as np

from .planet import _fit_ellipse_halir

def PLANET_2D_total(X, Y, TR, FAr, mask=None, pc_axis=-1):

    assert X.shape == Y.shape, 'X and Y must be the same shape!'
    X = np.moveaxis(X, pc_axis, -1)
    Y = np.moveaxis(Y, pc_axis, -1)
    sx, sy, npcs = X.shape[:]
    sh = X.shape[:2]

    # Mask can be used as Solution Mask, or just ones()
    if mask is None:
        mask = np.ones(sh, dtype=bool)

    # FAr can either be a scalar or array
    if not isinstance(type(FAr), np.ndarray):
        FAr = np.ones(sh)*FAr

    # Result arrays
    Phimap = np.zeros(sh)
    Mmap = np.zeros(sh)
    T1 = np.zeros(sh)
    T2 = np.zeros(sh)
    Amap = np.zeros(sh)
    Bmap = np.zeros(sh)
    Xcmap = np.zeros(sh)
    Ycmap = np.zeros(sh)

    # Loop over each pixel:
    for ii in range(sx):
        for jj in range(sy):

            # Flip angle for this pixel
            FA = FAr[ii, jj]

            # Skip empty phase-cycles
            if np.all(X[ii, jj, :] == 0) and np.all(Y[ii, jj, :]):
                T1[ii, jj] = -1
                T2[ii, jj] = -1
                continue

            # Fit the ellipse
            try:
                af = _fit_ellipse_halir(X[ii, jj, :], Y[ii, jj, :]).real
            except np.linalg.LinAlgError:
                T1[ii, jj] = -1
                T2[ii, jj] = -1
                continue

            if af.size != 6:
                # sometimes the fitting fails and returns nonsense...
                T1[ii, jj] = -1
                T2[ii, jj] = -1
                continue

            A, B, C, D, E, F = af[:]
            phi = 1/2 * np.arctan(B/(A - C))
            # phi = 1/2 * np.arctan2(B, A - C) # is this better?

            c = np.cos(phi)
            s = np.sin(phi)

            A1 = A*c**2 + B*c*s + C*s**2
            D1 = D*c + E*s
            E1 = E*c - D*s
            C1 = A*s**2 - B*c*s + C*c**2
            F11 = F - ((D1**2)/(4*A1) + (E1**2)/(4*C1))

            Xc = -D1/(2*A1)
            Yc = -E1/(2*C1)
            aa = np.sqrt(-F11/A1)
            bb = np.sqrt(-F11/C1)

            skip_redo = False
            if aa <= bb:
                if Xc >= 0:
                    skip_redo = True
                if Xc < 0:
                    phi -= np.pi*np.sign(phi)
            else:
                if Yc >= 0:
                    phi += np.pi/2
                else:
                    phi -= np.pi/2

            if not skip_redo:
                # If phi changed then we need to redo these
                c = np.cos(phi)
                s = np.sin(phi)
                A1 = A*c**2 + B*c*s + C*s**2
                D1 = D*c + E*s
                E1 = E*c - D*s
                C1 = A*s**2 - B*c*s + C*c**2
                F11 = F - ((D1**2)/(4*A1) + (E1**2)/(4*C1))

                Xc = -D1/(2*A1)
                Yc = -E1/(2*C1)
                aa = np.sqrt(-F11/A1)
                bb = np.sqrt(-F11/C1)


            if mask[ii, jj] == 1:
                # if FAr>LimitArray=acos(E1)=acos(exp(-TR/T1)):
                b = (-Xc*aa + bb*np.sqrt(Xc*Xc - aa*aa + bb*bb))/(Xc*Xc + bb*bb)
            else:
                # if FAr<=LimitArray=acos(E1)=acos(exp(-TR/T1)):
                b = ( Xc*aa + bb*np.sqrt(Xc*Xc - aa*aa + bb*bb))/(Xc*Xc + bb*bb)


            a = bb/(b*bb + Xc*np.sqrt(1 - b*b))
            M = Xc*(1 - b*b)/(1 - a*b)

            T2[ii, jj] = -TR/np.log(a)
            cFA = np.cos(FA)
            T1[ii, jj] = -TR/np.log(((a*(1 + cFA - a*b*cFA) - b)/(a*(1 + cFA - a*b) - b*cFA)))
            Phimap[ii, jj] = phi
            Mmap[ii, jj] = M
            Xcmap[ii, jj] = Xc
            Ycmap[ii, jj] = Yc
            Amap[ii, jj] = a
            Bmap[ii, jj] = b

    return(T1, T2, Phimap, Amap, Bmap, Xcmap, Ycmap, Mmap)

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from phantominator import shepp_logan
    from ssfp import bssfp

    # Shepp-Logan
    N, npcs = 128, 8
    M0, T1, T2 = shepp_logan((N, N, 1), MR=True, zlims=(-.25, .25))
    M0, T1, T2 = np.squeeze(M0), np.squeeze(T1), np.squeeze(T2)

    # Simulate bSSFP acquisition with linear off-resonance
    TR, alpha = 3e-3, np.deg2rad(15)
    pcs = np.linspace(0, 2*np.pi, npcs, endpoint=False)
    df, _ = np.meshgrid(
        np.linspace(-1/TR, 1/TR, N),
        np.linspace(-1/TR, 1/TR, N))
    sig = bssfp(T1, T2, TR, alpha, field_map=df, phase_cyc=pcs, M0=M0)

    # Do T1, T2 mapping for each pixel
    mask = np.abs(M0) > 1e-8

    # Make it noisy
    sig += 1e-7*(np.random.normal(0, 1, sig.shape) + 1j*np.random.normal(0, 1, sig.shape))*mask

    # Do the thing
    T1est, T2est, Phimap, Amap, Bmap, Xcmap, Ycmap, Mmap = PLANET_2D_total(sig.real, sig.imag, TR, alpha, mask, pc_axis=0)

    # Take a look
    plt.imshow(T1est*mask, vmin=0, vmax=np.max(T1.flatten()))
    plt.show()

    plt.imshow(T2est*mask, vmin=0, vmax=np.max(T2.flatten()))
    plt.show()
