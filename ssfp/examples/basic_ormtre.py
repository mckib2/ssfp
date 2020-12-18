'''Demonstrate basic ormtre usage.'''

import numpy as np
import matplotlib.pyplot as plt
from ssfp import bssfp, ormtre

if __name__ == '__main__':

    # Simulate some multi-TR acquisitions
    N = 128
    TR0, TR1 = 3e-3, 3.45e-3
    T1, T2, M0 = np.ones((N, N))*.8, np.ones((N, N))*.08, 1
    alpha = np.deg2rad(100)  # high flip angle
    dfTR = TR0
    df, _ = np.meshgrid(
        np.linspace(1/(2*dfTR), 1/dfTR, N),
        np.linspace(1/(2*dfTR), 1/dfTR, N))
    pcs0 = np.linspace(0, 2*np.pi, 4, endpoint=False)
    pcs1 = np.linspace(0, 2*np.pi, 2, endpoint=False)
    I0 = bssfp(T1, T2, TR0, alpha, df, phase_cyc=pcs0, M0=M0, target_pc_axis=-1)
    I1 = bssfp(T1, T2, TR1, alpha, df, phase_cyc=pcs0, M0=M0, target_pc_axis=-1)
    I2 = bssfp(T1, T2, TR1, alpha, df, phase_cyc=pcs1, M0=M0, target_pc_axis=-1)

    # Do the estimates
    theta_4_4 = ormtre(I0, I1, TR0, TR1)
    theta_4_2 = ormtre(I0, I2, TR0, TR1)

    nx, ny = 2, 3
    plt.subplot(nx, ny, 1)
    plt.imshow(df)
    plt.title('Truth')
    plt.xticks([], []), plt.yticks([], [])

    plt.subplot(nx, ny, 2)
    plt.imshow(theta_4_4)
    plt.title('4+4')
    plt.xticks([], []), plt.yticks([], [])

    plt.subplot(nx, ny, 3)
    plt.imshow((theta_4_4 - df)*100/df)
    plt.title('% diff')
    plt.xticks([], []), plt.yticks([], [])
    plt.colorbar()

    plt.subplot(nx, ny, 5)
    plt.imshow(theta_4_2)
    plt.title('4+2')
    plt.xticks([], []), plt.yticks([], [])

    plt.subplot(nx, ny, 6)
    plt.imshow((theta_4_2 - df)*100/df)
    plt.xticks([], []), plt.yticks([], [])
    plt.colorbar()


    plt.show()
