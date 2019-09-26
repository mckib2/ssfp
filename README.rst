SSFP
====

Simple steady-state free precession simulation.  The goal is to
provide a simple to use, pip-installable solution for simulating this
wonderful pulse sequence.

In this package:

- bSSFP: `bssfp()`

Installation
============

Should be as easy as:

.. code-block:: bash

    pip install ssfp

Usage
=====

See `ssfp.examples` for typical usage.  You can run examples like:

.. code-block:: bash

    python -m ssfp.examples.basic_bssfp

Balanced steady-state free precession can be called through `bssfp()`.
This is an implementation of equations [1--2] in [1]_.  These
equations are based on the Ernst-Anderson derivation [2]_ where
off-resonance is assumed to be subtracted as opposed to added (as in
the Freeman-Hill derivation [3]_).  Hoff actually gets Mx and My
flipped in the paper, so we fix that here.  We also assume that
the field map will be provided given the Freeman-Hill convention.

.. code-block:: python

    from ssfp import bssfp

    # Here's the simplest usage, see docstring for all the possible
    # function arguments
    sig = bssfp(T1, T2, TR, alpha)


References
==========
.. [1] Xiang, Qing‐San, and Michael N. Hoff. "Banding artifact
       removal for bSSFP imaging with an elliptical signal
       model." Magnetic resonance in medicine 71.3 (2014):
       927-933.
.. [2] Ernst, Richard R., and Weston A. Anderson. "Application of
       Fourier transform spectroscopy to magnetic resonance."
       Review of Scientific Instruments 37.1 (1966): 93-102.
.. [3] Freeman R, Hill H. Phase and intensity anomalies in
       fourier transform NMR. J Magn Reson 1971;4:366–383.
