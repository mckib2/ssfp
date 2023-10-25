"""Utility functions."""

import pathlib
import urllib.request
from math import ceil
import logging

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def ernst(TR: float, T1: np.ndarray) -> np.array:
    """Computes the Ernst angle.

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
    """

    # Don't divide by zero!
    alpha = np.zeros(T1.shape)
    idx = np.nonzero(T1)
    alpha[idx] = np.arccos(-TR/T1[idx])
    return alpha


def download_file(address: str, filename: str, force: bool=False) -> str:
    """Download a file into data folder.

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
    """

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
    if pbar is not None:
        pbar.close()
    return str(dest / filename)


class IndexTracker:
    """Use scroll wheel event to cycle through slices."""
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate slices')
        ax.set_xlabel(
            'left click to select point, right click to remove')

        self.X = X
        _rows, _cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        """Trigger scrolling event."""
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        """Load new slice."""
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


def choose_cntr(im: np.ndarray, slice_axis: int=-1):
    """Graphically choose point """

    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, np.moveaxis(im, slice_axis, -1))
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)

    # Get two clicks, first is the actual point, second
    # is just to close the input window
    cntr = fig.ginput(n=2, show_clicks=True)[0]

    # Get the current slice index and create cntr coord
    zidx = tracker.ind
    cntr = (zidx, int(cntr[0]), int(cntr[1]))
    plt.close(fig)

    logging.info('Choosing center point: %s', str(cntr))
    return cntr
