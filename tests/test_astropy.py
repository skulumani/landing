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
    images.getNearMSI()
    image_path = os.path.join(cwd, '../images')
    fit_files = [f for f in os.listdir(image_path) if f[-3:] == 'fit']

    test_image = 'm0157353946f4_2p_iof_dbl.fit'
    hdulist = fits.open(os.path.join(image_path, test_image))
    keys = hdulist[0].header.keys()
    hdulist.close()

    # useful keys for header
    xaxis_key = 'NAXIS1'
    yaxis_key = 'NAXIS2'
    expms_key = 'NEAR-010'
    teltemp_key = 'NEAR-015'
    ccdtemp_key = 'NEAR-016'
    sclkstart_key = 'NEAR-017'
    sclkmid_key = 'NEAR-018'
    # first element is the scalar. quat_key
    quat_key = ['NEAR-019', 'NEAR-020', 'NEAR-021', 'NEAR-022']

    def test_all_landing_images_same_header(self):
        for f in self.fit_files:
            data = fits.open(os.path.join(self.image_path, f))
            img_keys = data[0].header.keys()
            data.close()
            np.testing.assert_equal(img_keys, self.keys)

    def test_all_images_same_size(self):
        for f in self.fit_files:
            data = fits.open(os.path.join(self.image_path, f))
            header = data[0].header
            data.close()

            np.testing.assert_equal(header[self.xaxis_key], 537)
            np.testing.assert_equal(header[self.yaxis_key], 244)

    def test_inertial_state_against_spice(self):
        """Compare the state between the image and spice
        """
        spice.furnsh(self.near.metakernel)
        for f in self.fit_files:
            data = fits.open(os.path.join(self.image_path, f))
            header = data[0].header
            data.close()
            quat_fits = [header[x] for x in self.quat_key]
            sclk = header[self.sclkmid_key]
            et = spice.sct2e(int(self.near.near_id), sclk)

            Ri2b_fits = spice.q2m(quat_fits)
            pdb.set_trace()
            # get teh quaternion from teh kernels
            Ri2b_spice = spice.pxform(self.near.inertial_frame, 
                    self.near.near_body_frame, et)
            np.testing.assert_array_almost_equal(Ri2b_fits, Ri2b_spice)

        spice.kclear()
