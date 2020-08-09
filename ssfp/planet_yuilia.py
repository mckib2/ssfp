
import numpy as np

from .planet import _fit_ellipse_halir

class UniformArray:
    def __init__(self, val, sh):
        self.val = val
        self.shape = sh
    def cos_init(self):
        self.cos_val = np.cos(self.val)
    def __getitem__(self, *args):
        return self.val
    def cos(self):
        return self.cos_val


def planet(I, TR, FA, T1_guess=None, mask=None, pc_axis=-1):

    I = np.moveaxis(I, pc_axis, -1)
    X, Y = I.real, I.imag
    sx, sy, npcs = I.shape[:]
    sh = I.shape[:2]

    # FA can either be a scalar or array;
    # if it's a scalar, make it look like an array
    if not isinstance(type(FA), np.ndarray):
        FA = UniformArray(FA, sh)
        FA.cos_init()
        # F[ii, jj, ...] returns FA for any arguments to []
        # np.cos(F[ii, jj, ...]) returns np.cos(FA) for any arguments to []

    # Choose an intial estimate for T1
    if T1_guess is None:
        T1_guess = UniformArray(1, sh) # 1sec arbitrariliy

    # Choose which pixels are reconstructed
    if mask is None:
        mask = UniformArray(True, sh)

    # Result arrays
    Phimap = np.empty(sh)
    Mmap = np.empty(sh)
    T1 = np.empty(sh)
    T2 = np.empty(sh)
    Amap = np.empty(sh)
    Bmap = np.empty(sh)
    Xcmap = np.empty(sh)
    Ycmap = np.empty(sh)

    # Loop over each pixel:
    for ii in range(sx):
        for jj in range(sy):

            # Skip masked items
            if not mask[ii, jj]:
                continue

            # Skip empty phase-cycles
            if np.all(I[ii, jj, :] == 0):
                T1[ii, jj] = -1
                T2[ii, jj] = -1
                continue

            # Fit the ellipse
            try:
                af = _fit_ellipse_halir(
                    I[ii, jj, :].real,
                    I[ii, jj, :].imag).real
            except np.linalg.LinAlgError:
                T1[ii, jj] = -1
                T2[ii, jj] = -1
                continue

            if af.size != 6:
                # sometimes the fitting fails and returns nonsense
                T1[ii, jj] = -1
                T2[ii, jj] = -1
                continue

            A, B, C, D, E, F = af[:]
            phi = 1/2 * np.arctan(B/(A - C))
            # phi = 1/2 * np.arctan2(B, A - C) # is this better?

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
                # If phi changed then we need to recompute
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

            # Decide sign of first term of b
            if FA[ii, jj] > np.arccos(np.exp(-TR/T1_guess[ii, jj])):
                bsign = -1
            else:
                bsign = 1

            b = (bsign*Xc*aa + bb*np.sqrt(Xc*Xc - aa*aa + bb*bb))/(Xc*Xc + bb*bb)
            a = bb/(b*bb + Xc*np.sqrt(1 - b*b))
            M = Xc*(1 - b*b)/(1 - a*b)

            T2[ii, jj] = -TR/np.log(a)
            cFA = np.cos(FA[ii, jj])
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
    T1est, T2est, Phimap, Amap, Bmap, Xcmap, Ycmap, Mmap = planet(sig, TR, alpha, mask, pc_axis=0)

    # Take a look
    plt.imshow(T1est*mask, vmin=0, vmax=np.max(T1.flatten()))
    plt.show()

    plt.imshow(T2est*mask, vmin=0, vmax=np.max(T2.flatten()))
    plt.show()
