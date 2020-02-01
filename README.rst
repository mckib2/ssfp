SSFP
====

Simple steady-state free precession simulation.  The goal is to
provide a simple to use, pip-installable solution for simulating and
working with this wonderful pulse sequence.

In this package:

- bSSFP: `bssfp()`
- GS solution: `gs_recon()`
- PLANET: `planet()`
- 3D Region Growing Phase Correction: `rgphcorr3d()`
- Robust Coil Combination: `robustcc()`

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

We can also easily get the Geometric Solution to the elliptical
signal model as described in [1]_ as follows:

.. code-block:: python

    from ssfp import gs_recon
    recon = gs_recon(phased_cycled_images, pc_axis=-1)

    # Notice that we can specify the axis where the phase-cycles live

PLANET [4]_ is a method for simultaneous T1, T2 fitting for bSSFP
phase-cycled data.  Call like this:

.. code-block:: python

    from ssfp import planet

    # For a single pixel:
    Meff, T1, T2 = planet(
        phased_cycled_pixels, alpha, TR, T1_guess,
        pcs=np.deg2rad([0, 90, 180, 270, etc...]))

3D Region Growing Phase Correction [5]_ is an algorithm for
determining water and fat images from a single bSSFP acquisition.
It can be called like this:

.. code-block:: python

    from ssfp import rgphcorr3d
    phase_corrected = rgphcorr3d(
        dataset3d, cellsize=(4, 4, 4), slice_axis=-1)

    # see ssfp.examples.basic_rgphcorr for full usage example

Robust Coil Combination for bSSFP Elliptical Signal Model [6]_ is a
coil combination method that preserves the elliptical relationships
between phase-cycled pixels.  It has two variants: simple and full.
By default, the simple method is called.  The full method is very
slow and only used for validation of the simple method.
Robust coil combination can be called like this:

.. code-block:: python

    from ssfp import robustcc

    sx, sy, sz, num_pc, num_coils = data.shape[:]
    coil_combined = robustcc(data, pc_axis=-2, coil_axis=-1)

    # see ssfp.examples.basic_robustcc for more usage examples

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
.. [4] Shcherbakova, Yulia, et al. "PLANET: an ellipse fitting
       approach for simultaneous T1 and T2 mapping using
       phase‐cycled balanced steady‐state free precession."
       Magnetic resonance in medicine 79.2 (2018): 711-722.
.. [5] Hargreaves, Brian A., et al. "Fat‐suppressed steady‐state
       free precession imaging using phase detection." Magnetic
       Resonance in Medicine: An Official Journal of the
       International Society for Magnetic Resonance in Medicine
       50.1 (2003): 210-213.
.. [6] N. McKibben, G. Tarbox, E. DiBella, and N. K. Bangerter,
       "Robust Coil Combination for bSSFP Elliptical Signal
       Model," Proceedings of the 28th Annual Meeting of the
       ISMRM; Sydney, NSW, Australia, 2020.
