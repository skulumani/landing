"""Test SPICEYPY
"""

import spiceypy as spice
import numpy as np
from urllib import urlretrieve
import os
import pdb

true_initial_pos = [-5461446.61080924 ,-4434793.40785864 ,-1200385.93315424]

def test_spiceypy_installation_correct():
    spice_version = 'CSPICE_N0065'
    np.testing.assert_equal(spice.tkvrsn('TOOLKIT'),spice_version)

def test_spiceypy_cassini(tmpdir):
    data_path = tmpdir.mkdir("data")
    kernel_urls = [
            'http://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/a_old_versions/naif0009.tls',
            'http://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/sclk/cas00084.tsc',
            'http://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/pck/cpck05Mar2004.tpc',
            'http://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/fk/release.11/cas_v37.tf',
            'http://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/ck/04135_04171pc_psiv2.bc',
            'http://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/spk/030201AP_SK_SM546_T45.bsp',
            'http://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/ik/release.11/cas_iss_v09.ti',
            'http://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/spk/020514_SE_SAT105.bsp',
            'http://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/spk/981005_PLTEPH-DE405S.bsp'
            ]

    for url in kernel_urls:
        filename = url.split('/')[-1]
        fullfilename = os.path.join(data_path.dirname, filename)
        urlretrieve(url, fullfilename)
        spice.furnsh(fullfilename)

    step = 4000
    utc = ['Jun 20, 2004', 'Dec 1, 2005']

    etOne = spice.str2et(utc[0])
    etTwo = spice.str2et(utc[1])
    print("ET One: {}, ET Two: {}".format(etOne, etTwo))

    times = [x*(etTwo-etOne)/step + etOne for x in range(step)]

    positions, lightTimes = spice.spkpos('Cassini', times, 'J2000', 'None', 'SATURN BARYCENTER')
    pdb.set_trace()
    np.testing.assert_array_almost_equal(positions[0], true_initial_pos)
