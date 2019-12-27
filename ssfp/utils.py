'''Utility functions.'''

import numpy as np

def ernst(TR, T1):
    '''Computes the Ernst angle.

    Parameters
    ----------
    TR : float
        repetition time.
    T1 : array_like
        longitudinal exponential decay time constant.

    Returns
    -------
    alpha : array_like
        Ernst angle in rad.

    Notes
    -----
    Implements equation [14.9] from [1]_.

    References
    ----------
    .. [1] Notes from Bernstein, M. A., King, K. F., & Zhou, X. J.
           (2004). Handbook of MRI pulse sequences. Elsevier.
    '''

    # Don't divide by zero!
    alpha = np.zeros(T1.shape)
    idx = np.nonzero(T1)
    alpha[idx] = np.arccos(-TR/T1[idx])
    return alpha
