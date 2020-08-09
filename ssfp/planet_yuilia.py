
import numpy as np

#from .planet import _fit_ellipse_halir

class UniformArray:
    def __init__(self, val, sh):
        self.val = val
        self.shape = sh
    def flatten(self):
        return self
    def cos_init(self):
        self.cos_val = np.cos(self.val)
    def __getitem__(self, *args):
        return self.val
    def cos(self):
        return self.cos_val

def _fit_ellipse_halir(I):

    # quadratic pt of design matrix
    D1 = np.stack((I.real**2, I.real*I.imag, I.imag**2), axis=1).transpose((0, 2, 1))
    # lin part design matrix
    D2 = np.stack((I.real, I.imag, np.ones(I.shape)), axis=1).transpose((0, 2, 1))

    # quadratic part of the scatter matrix
    S1 = np.einsum('fji,fjk->fik', D1, D1)
    # combined part of the scatter matrix
    S2 = np.einsum('fji,fjk->fik', D1, D2)
    # linear part of the scatter matrix
    S3 = np.einsum('fji,fjk->fik', D2, D2)

    # for getting a2 from a1
    T = np.einsum('fij,fkj->fik', -1*np.linalg.pinv(S3), S2)

    # reduced scatter matrix; premult by C1^-1
    M = S1 + np.einsum('fij,fjk->fik', S2, T)
    M = np.stack((M[:, 2, :]/2, -1*M[:, 1, :], M[:, 0, :]/2), axis=1)
    # solve eigensystem
    _eval, evec = np.linalg.eig(M)

    a1 = np.empty((M.shape[0], 3))
    for ii in range(M.shape[0]):
        # evaluate a’Ca
        cond = 4*evec[ii, 0, :]*evec[ii, 2, :] - evec[ii, 1, :]**2
        # eigenvector for min. pos. eigenvalue
        if not np.sum(cond > 0):
            # Failed to fit the ellipse! send back 0s
            a1[ii, :] = 0
        else:
            a1[ii, :] = evec[ii, :, cond > 0]

    # ellipse coefficients
    a = np.concatenate((a1, np.einsum('fij,fj->fi', T, a1)), axis=-1)
    return a/np.linalg.norm(a)


def planet(I, TR, FA, T1_guess=None, mask=None, pc_axis=-1):

    I = np.moveaxis(I, pc_axis, -1)
    X, Y = I.real, I.imag
    sx, sy, npcs = I.shape[:]
    sh = I.shape[:2]

    # FA can either be a scalar or array;
    # Choose an intial estimate for T1
    if T1_guess is None:
        T1_guess = 1 # 1sec arbitrariliy

    # Choose which pixels are reconstructed
    if mask is None:
        mask = UniformArray(True, sh)

    # Fit all ellipses that are nonzero
    if mask is None:
        recon_idx = np.nonzero(np.sum(np.abs(I).reshape((-1, npcs)), axis=-1))[0]
    else:
        recon_idx = np.nonzero(mask.flatten())[0]
    recon_idx = list(range(np.prod(sh)))
    ellipse_coefs = _fit_ellipse_halir(np.take(
        I.reshape((-1, npcs)),
        recon_idx,
        axis=0))

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
    if isinstance(FA, np.ndarray):
        bsign = np.ones(FA.shape)
        bsign[FA > np.arccos(np.exp(-TR/T1_guess))] = -1
    else:
        if FA > np.arccos(np.exp(-TR/T1_guess)):
            bsign = -1
        else:
            bsign = 1

    # Compute interesting values
    Xc2 = Xc*Xc
    b = (bsign*Xc*aa + bb*np.sqrt(Xc2 - aa*aa + bb*bb))/(Xc2 + bb*bb)
    a = bb/(b*bb + Xc*np.sqrt(1 - b*b))
    M = Xc*(1 - b*b)/(1 - a*b)

    T2 = np.zeros(np.prod(sh))
    T2[recon_idx] = -TR/np.log(a)

    T1 = np.zeros(np.prod(sh))
    cFA = np.cos(FA)
    T1[recon_idx] = -TR/np.log(((a*(1 + cFA - a*b*cFA) - b)/(a*(1 + cFA - a*b) - b*cFA)))

    # Pack the rest of the result matrices
    Phimap = np.zeros(np.prod(sh))
    Phimap[recon_idx] = phi
    Mmap = np.zeros(np.prod(sh))
    Mmap[recon_idx] = M
    Xcmap = np.zeros(np.prod(sh))
    Xcmap[recon_idx] = Xc
    Ycmap = np.zeros(np.prod(sh))
    Ycmap[recon_idx] = Yc
    Amap = np.zeros(np.prod(sh))
    Amap[recon_idx] = a
    Bmap = np.zeros(np.prod(sh))
    Bmap[recon_idx] = b

    return(
        np.reshape(T1, sh),
        np.reshape(T2, sh),
        np.reshape(Phimap, sh),
        np.reshape(Amap, sh),
        np.reshape(Bmap, sh),
        np.reshape(Xcmap, sh),
        np.reshape(Ycmap, sh),
        np.reshape(Mmap, sh),
    )

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
    T1est, T2est, Phimap, Amap, Bmap, Xcmap, Ycmap, Mmap = planet(sig, TR, alpha, mask=mask, pc_axis=0)

    # Take a look
    plt.imshow(T1est*mask, vmin=0, vmax=np.max(T1.flatten()))
    plt.show()

    plt.imshow(T2est*mask, vmin=0, vmax=np.max(T2.flatten()))
    plt.show()
