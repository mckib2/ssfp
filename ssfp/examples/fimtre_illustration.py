"""Picture how FIMTRE works."""

import numpy as np
import matplotlib.pyplot as plt

from ssfp import bssfp, fimtre, gs_recon
from ellipsinator import rotate_points


if __name__ == '__main__':
    pbssfp = lambda TR, pcs: bssfp(T1, T2, TR=TR, alpha=alpha, field_map=df, phase_cyc=pcs)
    T1, T2 = 2, 1
    TRs = [3e-3, 5e-3]
    alpha = np.deg2rad(87)
    pcs2 = np.linspace(0, 2*np.pi, 2, endpoint=False)
    pcs4 = np.linspace(0, 2*np.pi, 4, endpoint=False)
    many_pcs = np.linspace(0, 2*np.pi, 200, endpoint=True)
    df = 1/(2*TRs[0])
    e1 = pbssfp(TRs[0], pcs4)
    e1_ext = pbssfp(TRs[0], many_pcs)
    e2 = pbssfp(TRs[1], pcs2)
    e2_ext = pbssfp(TRs[1], many_pcs)
    df_est = fimtre(e1, e2, *TRs, rad=False)
    rad_est = fimtre(e1, e2, *TRs, rad=True)

    plt.plot(e1_ext.real, e1_ext.imag, 'b', label='TR1')
    plt.plot(e1.real, e1.imag, 'bx')
    plt.plot(e2_ext.real, e2_ext.imag, 'r', label='TR2')
    plt.plot(e2.real, e2.imag, 'rx')

    # Annotations
    plt.plot(e2.real, e2.imag, 'r--')
    xr, yr = rotate_points(e2.real, e2.imag, phi=rad_est)
    plt.plot(xr, yr, 'rx')
    plt.plot(xr, yr, 'r--')
    ctr0 = gs_recon(np.atleast_2d(e1), pc_axis=-1, second_pass=False)
    plt.plot(ctr0.real, ctr0.imag, 'bo')
    
    plt.title(f'True: {df:.2f}Hz, Est: {df_est[0]:.2f}Hz')
    plt.legend()
    plt.xlabel('Real (a.u.)')
    plt.ylabel('Imag (a.u.)')
    plt.axis('square')
    plt.show()
