"""Test SPICEYPY
"""

import spiceypy as spice
import numpy as np
from urllib import urlretrieve
import os
import pdb

from spice_test import download_cassini_spice

true_initial_pos = [-5461446.61080924 ,-4434793.40785864 ,-1200385.93315424]

def test_spiceypy_installation_correct():
    spice_version = 'CSPICE_N0065'
    np.testing.assert_equal(spice.tkvrsn('TOOLKIT'),spice_version)

def test_spiceypy_cassini(tmpdir):
    pdb.set_trace()
    download_cassini_spice()
    step = 4000
    utc = ['Jun 20, 2004', 'Dec 1, 2005']

    etOne = spice.str2et(utc[0])
    etTwo = spice.str2et(utc[1])
    print("ET One: {}, ET Two: {}".format(etOne, etTwo))

    times = [x*(etTwo-etOne)/step + etOne for x in range(step)]

    positions, lightTimes = spice.spkpos('Cassini', times, 'J2000', 'None', 'SATURN BARYCENTER')
    pdb.set_trace()
    np.testing.assert_array_almost_equal(positions[0], true_initial_pos)
