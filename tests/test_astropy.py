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
    np.testing.assert_equal(astropy.__version__, '1.3.3')


class TestNEARFitsImages():
    # make sure all the images are downloaded
    near = kernels.NearKernels()
    near_images = images.NearImages(near)
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
        pdb.set_trace()
        for image in self.near_images.images:
            R_i2b_fits = spice.q2m(image['quat_i2b'])
            R_i2b_spice = image['spice_R_i2b']
            np.testing.assert_array_almost_equal(R_i2b_fits, R_i2b_spice, 
                    decimal=1)

