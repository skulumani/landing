"""Test out astropy
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import astropy
from astropy.io import fits
import numpy as np

from .. import images

cwd = os.path.realpath(os.path.dirname(__file__))
def test_astropy_installed():
    np.testing.assert_equal(astropy.__version__, '1.3.3')

class TestNEARFitsImages():
    # make sure all the images are downloaded
    images.getNearMSI()
    image_path = os.path.join(cwd, '../images') 
    fit_files = [f for f in os.listdir(self.image_path) if f[-3:] == 'fit']

    test_image = 'm0157353946f4_2p_iof_dbl.fit'
    hdulist = fits.open(os.path.join(image_path,test_image))
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
            data = fits.open(os.path.join(self.image_path,f))
            img_keys = data[0].header.keys()
            data.close()
            np.testing.assert_equal(img_keys, self.keys)

    def test_image_exposure_clock_timing(self):
        for f in self.fit_files:
            data = fit.open(os.path.join(self.image_path, f))
            header = data[0].header

            
