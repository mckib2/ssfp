"""Basic usage of bSSFP."""

import numpy as np
import matplotlib.pyplot as plt
from ssfp import bssfp


if __name__ == '__main__':
    M0, T1, T2 = 1, 1, .5
    TR, alpha = 3e-3, np.deg2rad(30)
    field_map = 1/(2*TR)
    pcs = np.linspace(0, 2*np.pi, 100, endpoint=False)
    sig = bssfp(
        T1, T2, TR, alpha, field_map, phase_cyc=pcs, M0=M0,
        delta_cs=0, phi_rf=0, phi_edd=0, phi_drift=0)

    fig, ax1 = plt.subplots()
    dpcs = np.rad2deg(pcs)
    ax1.plot(dpcs, np.abs(sig), 'k-')
    ax1.set_ylabel('Magnitude (a.u.)')
    ax1.set_xlabel('Phase Cycle (in deg)')

    ax2 = ax1.twinx()
    ax2.plot(dpcs, np.angle(sig), 'k--')
    ax2.set_ylabel('Phase (rad)')

    plt.title('bSSFP Spectrum')
    plt.show()
