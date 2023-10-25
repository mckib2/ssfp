"""Show basic usage of FIMTRE."""

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import normalized_root_mse
from phantominator import shepp_logan

from ssfp import bssfp, fimtre, planet


if __name__ == '__main__':

    fimtre_results = 1
    planet_results = 0
    sigma = 4e-5
    resid_mult = 10

    # Shepp-Logan
    N, nslices = 256, 1
    M0, T1, T2 = shepp_logan((N, N, nslices), MR=True, zlims=(-.25, 0))
    M0, T1, T2 = np.squeeze(M0), np.squeeze(T1), np.squeeze(T2)
    mask = np.abs(M0) > 1e-8

    # Simulate bSSFP acquisition with linear off-resonance
    TR0, TR1 = 3e-3, 6e-3
    alpha = np.deg2rad(80)
    alpha_lo = np.deg2rad(12)
    pcs8 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    pcs6 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    pcs4 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    pcs2 = np.linspace(0, 2 * np.pi, 2, endpoint=False)
    df, _ = np.meshgrid(
        np.linspace(-1 / (2 * TR1), 1 / (2 * TR1), N),
        np.linspace(-1 / (2 * TR1), 1 / (2 * TR1), N))
    df *= mask

    if fimtre_results:
        I0_hi = bssfp(T1, T2, TR0, alpha, field_map=df,
                      phase_cyc=pcs4[None, None, :], M0=M0)
        I1_6_hi = bssfp(T1, T2, TR1, alpha, field_map=df,
                        phase_cyc=pcs2[None, None, :], M0=M0)
        I1_8_hi = bssfp(T1, T2, TR1, alpha, field_map=df,
                        phase_cyc=pcs4[None, None, :], M0=M0)
        I0_lo = bssfp(T1, T2, TR0, alpha_lo, field_map=df,
                      phase_cyc=pcs4[None, None, :], M0=M0)
        I1_6_lo = bssfp(T1, T2, TR1, alpha_lo, field_map=df,
                        phase_cyc=pcs2[None, None, :], M0=M0)
        I1_8_lo = bssfp(T1, T2, TR1, alpha_lo, field_map=df,
                        phase_cyc=pcs4[None, None, :], M0=M0)
    if planet_results:
        Ip6_hi = bssfp(T1, T2, TR0, alpha, field_map=df,
                       phase_cyc=pcs6[None, None, :], M0=M0)
        Ip8_hi = bssfp(T1, T2, TR0, alpha, field_map=df,
                       phase_cyc=pcs8[None, None, :], M0=M0)
        Ip6_lo = bssfp(T1, T2, TR0, alpha_lo, field_map=df,
                       phase_cyc=pcs6[None, None, :], M0=M0)
        Ip8_lo = bssfp(T1, T2, TR0, alpha_lo, field_map=df,
                       phase_cyc=pcs8[None, None, :], M0=M0)

    # Make it noisy
    np.random.seed(0)
    if fimtre_results:
        I0_hi += sigma * (np.random.normal(0, 1, I0_hi.shape) +
                          1j * np.random.normal(0, 1, I0_hi.shape))
        I1_6_hi += sigma * (np.random.normal(0, 1, I1_6_hi.shape) +
                            1j * np.random.normal(0, 1, I1_6_hi.shape))
        I1_8_hi += sigma * (np.random.normal(0, 1, I1_8_hi.shape) +
                            1j * np.random.normal(0, 1, I1_8_hi.shape))
        I0_lo += sigma * (np.random.normal(0, 1, I0_lo.shape) +
                          1j * np.random.normal(0, 1, I0_lo.shape))
        I1_6_lo += sigma * (np.random.normal(0, 1, I1_6_lo.shape) +
                            1j * np.random.normal(0, 1, I1_6_lo.shape))
        I1_8_lo += sigma * (np.random.normal(0, 1, I1_8_lo.shape) +
                            1j * np.random.normal(0, 1, I1_8_lo.shape))
    if planet_results:
        Ip6_hi += sigma * (np.random.normal(0, 1, Ip6_hi.shape) +
                           1j * np.random.normal(0, 1, Ip6_hi.shape))
        Ip8_hi += sigma * (np.random.normal(0, 1, Ip8_hi.shape) +
                           1j * np.random.normal(0, 1, Ip8_hi.shape))
        Ip6_lo += sigma * (np.random.normal(0, 1, Ip6_lo.shape) +
                           1j * np.random.normal(0, 1, Ip6_lo.shape))
        Ip8_lo += sigma * (np.random.normal(0, 1, Ip8_lo.shape) +
                           1j * np.random.normal(0, 1, Ip8_lo.shape))

    # Do the thing
    if fimtre_results:
        theta6_hi = fimtre(I0_hi, I1_6_hi, TR0, TR1, rad=False) * mask
        theta8_hi = fimtre(I0_hi, I1_8_hi, TR0, TR1, rad=False) * mask
        theta6_lo = fimtre(I0_lo, I1_6_lo, TR0, TR1, rad=False) * mask
        theta8_lo = fimtre(I0_lo, I1_8_lo, TR0, TR1, rad=False) * mask

        # reverse polarity if it makes sense
        if normalized_root_mse(theta6_hi, df) > normalized_root_mse(-1 * theta6_hi, df):
            theta6_hi *= -1
        if normalized_root_mse(theta8_hi, df) > normalized_root_mse(-1 * theta8_hi, df):
            theta8_hi *= -1
        if normalized_root_mse(theta6_lo, df) > normalized_root_mse(-1 * theta6_lo, df):
            theta6_lo *= -1
        if normalized_root_mse(theta8_lo, df) > normalized_root_mse(-1 * theta8_lo, df):
            theta8_lo *= -1

    if planet_results:
        _Meff, _T1, _T2, theta_planet6_hi = planet(Ip6_hi, TR=TR0, alpha=alpha, pc_axis=-1) * mask
        _Meff, _T1, _T2, theta_planet8_hi = planet(Ip8_hi, TR=TR0, alpha=alpha, pc_axis=-1) * mask
        _Meff, _T1, _T2, theta_planet6_lo = planet(Ip6_lo, TR=TR0, alpha=alpha_lo, pc_axis=-1) * mask
        _Meff, _T1, _T2, theta_planet8_lo = planet(Ip8_lo, TR=TR0, alpha=alpha_lo, pc_axis=-1) * mask

    vmn, vmx = np.min(df.flatten()), np.max(df.flatten())
    opts = {'vmin': vmn, 'vmax': vmx}  # , 'cmap': 'hot'}
    nx, ny = fimtre_results * 4 + planet_results * 4, 3
    idx = 1
    plt.subplot(nx, ny, idx)
    plt.imshow(df, **opts)
    plt.title('True off-res')
    plt.tick_params(axis='both', labelsize=0, length=0)
    idx += 1

    _wrote_est_hdr = False


    def _imshow(est, title, idx0):
        global _wrote_est_hdr
        plt.subplot(nx, ny, idx0)
        plt.imshow(est, **opts)
        plt.ylabel(title)
        plt.tick_params(axis='both', labelsize=0, length=0)
        if not _wrote_est_hdr:
            plt.title('Estimates')
            _wrote_est_hdr = True
        idx0 += 1
        return idx0


    _wrote_diff_hdr = False


    def _diffim(est, idx0):
        global _wrote_diff_hdr
        plt.subplot(nx, ny, idx0)
        plt.imshow(np.abs(df - est) * resid_mult, **opts)
        plt.annotate(f'NRMSE: {normalized_root_mse(df, est):g}', (.05, .05), xycoords='axes fraction', color='k')
        plt.tick_params(axis='both', labelsize=0, length=0)
        if not _wrote_diff_hdr:
            plt.title(f'Residual (x{resid_mult})')
            _wrote_diff_hdr = True
        idx0 += 2
        return idx0


    if fimtre_results:
        idx = _imshow(theta6_hi, 'FIMTRE 6 PCs (high FA)', idx)
        idx = _diffim(theta6_hi, idx)

        idx = _imshow(theta8_hi, 'FIMTRE 8 PCs (high FA)', idx)
        idx = _diffim(theta8_hi, idx)

        idx = _imshow(theta6_lo, 'FIMTRE 6 PCs (low FA)', idx)
        idx = _diffim(theta6_lo, idx)

        idx = _imshow(theta8_lo, 'FIMTRE 8 PCs (low FA)', idx)
        idx = _diffim(theta8_lo, idx)

    if planet_results:
        idx = _imshow(theta_planet6_hi, 'PLANET 6 PCs (high FA)', idx)
        idx = _diffim(theta_planet6_hi, idx)

        idx = _imshow(theta_planet8_hi, 'PLANET 8 PCs (high FA)', idx)
        idx = _diffim(theta_planet8_hi, idx)

        idx = _imshow(theta_planet6_lo, 'PLANET 6 PCs (low FA)', idx)
        idx = _diffim(theta_planet6_lo, idx)

        idx = _imshow(theta_planet8_lo, 'PLANET 8 PCs (low FA)', idx)
        idx = _diffim(theta_planet8_lo, idx)

    plt.show()
