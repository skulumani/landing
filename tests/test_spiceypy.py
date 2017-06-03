"""Test SPICEYPY
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import spiceypy as spice
import numpy as np
from urllib import urlretrieve
import os
import pdb

from .. import kernels
def test_spiceypy_installation_correct():
    spice_version = 'CSPICE_N0065'
    np.testing.assert_equal(spice.tkvrsn('TOOLKIT'),spice_version)

class TestSpiceyPyFunctions():
    cass = kernels.CassiniKernels()
    spice.furnsh(cass.metakernel) 
    utc = ['Jun 20, 2004', 'Dec 1, 2005']

    etOne = spice.str2et(utc[0])
    etTwo = spice.str2et(utc[1])
    step = 4000

    times = [x*(etTwo-etOne)/step + etOne for x in range(step)]

    def test_spiceypy_cassini(self):
        true_initial_pos = [-5461446.61080924 ,-4434793.40785864 ,-1200385.93315424]
        positions, lightTimes = spice.spkpos('Cassini', self.times, 'J2000', 'None', 'SATURN BARYCENTER')
        np.testing.assert_array_almost_equal(positions[0], true_initial_pos)

    def test_spicepy_rotation_matrix_identity(self):
        R_E2E = spice.pxform('IAU_EARTH', 'IAU_EARTH', self.etOne)
        np.testing.assert_array_almost_equal(R_E2E, np.eye(3))

    def test_spicepy_state_transformation(self):
        T = spice.sxform('IAU_EARTH', 'IAU_SATURN', self.etOne)
        R = spice.pxform('IAU_EARTH', 'IAU_SATURN', self.etOne)
        (Rout, wout) = spice.xf2rav(T)
        np.testing.assert_array_almost_equal(Rout, R)

class TestNEARKernels():
    near = kernels.NearKernels()
    kernels.getKernels(near)
    metakernel = kernels.writeMetaKernel(near, 'near2001.tm')

    spice.furnsh(metakernel)
    # ckid = spice.ckobj(near.Ck)[0]
    # cover = spice.ckcov(near.Ck, ckid, False, 'INTERVAL', 0.0, 'SCLK')
    
#    def test_near_body_frames(self):
#        """Transformation from Body fixed frame to prime frame
# 
#        There is a constant rotation of 135 deg about the Z/Third axis
#        """
#        R, av, clkout = spice.ckgpav(self.ckid, self.cover[0], 0, 'NEAR_SC_BUS')
#        ang = 135*np.pi/180
#        R_act = np.array([[np.cos(ang), -np.sin(ang), 0], 
#                          [np.sin(ang), np.cos(ang), 0],
#                          [0, 0, 1]])
# 
#        np.testing.assert_array_almost_equal(R, R_act)
#        np.testing.assert_array_almost_equal(av, np.zeros(3))
#        np.testing.assert_almost_equal(clkout, self.cover[0])
