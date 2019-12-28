'''Utility functions.'''

import pathlib
import urllib.request
from math import ceil
import logging

import numpy as np
from tqdm import tqdm

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

def download_file(address, filename, force=False):
    '''Download a file into data folder.

    Parameters
    ----------
    address : str
        URL to get the file.
    filename : str
        Filename to save the file at URL to.
    force : bool, optional
        Download again, even if the file already exists locally.

    Returns
    -------
    path : str
        Local path to downloaded file.
    '''

    # Make sure destination directory exists
    dest = pathlib.Path('data/')
    dest.mkdir(parents=True, exist_ok=True)

    # Don't download if we already have it
    if not force and (dest / filename).exists():
        logging.info('File is already downloaded!')
        return str(dest / filename)

    # Else, get file from interwebs
    pbar = None
    def _progress(_num_blocks, block_size, file_size):
        nonlocal pbar
        if pbar is None:
            pbar = tqdm(
                desc='Downloading file',
                total=ceil(file_size/block_size), leave=False)
        pbar.update(1)

    _local_filename, _ = urllib.request.urlretrieve(
        address, str(dest / filename), _progress)
    pbar.close()
    return str(dest / filename)
