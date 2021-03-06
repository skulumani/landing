"""Test out astropy
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import astropy
from astropy.io import fits
import numpy as np
import spiceypy as spice

from .. import images
from .. import kernels

import pdb

cwd = os.path.realpath(os.path.dirname(__file__))


def test_astropy_installed():
    np.testing.assert_equal(astropy.__version__, '2.0.2')


class TestNEARFitsImages():
    # make sure all the images are downloaded
    near = kernels.NearKernels()
    near_images = images.NearImages(near=near, downloaded=True)
    image_path = os.path.join(cwd, '../images')

    # keys to test for
    xaxis_key = 'NAXIS1'
    yaxis_key = 'NAXIS2'

    def test_all_images_same_size(self):
        for f in self.near_images.fits:
            data = fits.open(os.path.join(self.image_path, f))
            header = data[0].header
            data.close()

            np.testing.assert_equal(header[self.xaxis_key], 537)
            np.testing.assert_equal(header[self.yaxis_key], 244)

    def test_fits_image_attitude_compared_to_spice(self):
        spice.furnsh(self.near.metakernel)
        for image in self.near_images.images:
            q_i2b_fits = image['quat_i2b']
            q_i2b_spice = spice.m2q(image['spice_R_i2b'])
            np.testing.assert_array_almost_equal(q_i2b_fits, q_i2b_spice, 
                    decimal=1)
        
        spice.kclear()

    def test_fits_image_sun_position_compared_to_spice(self):
        spice.furnsh(self.near.metakernel)
        for image in self.near_images.images:
            sun2sc_pos_fits = [spice.convrt(x, 'KM', 'AU') for x in
                    image['inertial_sun_pos']]
            sun2sc_pos_spice = [spice.convrt(x, 'KM', 'AU') for x in 
                    image['spice_inertial_sun_pos']]
            np.testing.assert_array_almost_equal(sun2sc_pos_fits, 
                    sun2sc_pos_spice, decimal=3)

        spice.kclear()

    def test_fits_image_eros_position_compared_to_spice(self):
        spice.furnsh(self.near.metakernel)
        for image in self.near_images.images:
            sun2eros_pos_fits = [spice.convrt(x, 'KM', 'AU') for x in
                    image['inertial_target_pos']]
            sun2eros_pos_spice = [spice.convrt(x, 'KM', 'AU') for x in
                    image['spice_inertial_target_pos']]
            np.testing.assert_array_almost_equal(sun2eros_pos_fits, 
                    sun2eros_pos_spice, decimal=3)

        spice.kclear()

    def test_fits_image_eros_to_sc_position_compare_to_spice(self):
        spice.furnsh(self.near.metakernel)
        for image in self.near_images.images:
            eros2sc_pos_fits = image['inertial_eros2sc_pos']
            eros2sc_pos_spice = image['spice_inertial_eros2sc_pos']
            np.testing.assert_array_almost_equal(eros2sc_pos_fits, 
                    eros2sc_pos_spice, decimal=0)

        spice.kclear()
