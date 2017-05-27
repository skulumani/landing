"""Test the Python SPICE functionality by plotting NEAR

"""

import spiceypy as spice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from urllib import urlretrieve
import os
import pdb


def download_spice():
    if not os.path.isdir('./cassini'):
        os.mkdir('cassini')

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
            fullfilename = os.path.join('cassini', filename)
            urlretrieve(url, fullfilename)
            spice.furnsh(fullfilename)

if __name__=='__main__':
    download_spice()
    step = 4000
    utc = ['Jun 20, 2004', 'Dec 1, 2005']

    etOne = spice.str2et(utc[0])
    etTwo = spice.str2et(utc[1])
    print("ET One: {}, ET Two: {}".format(etOne, etTwo))

    times = [x*(etTwo-etOne)/step + etOne for x in range(step)]

    positions, lightTimes = spice.spkpos('Cassini', times, 'J2000', 'None', 'SATURN BARYCENTER')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(positions[:,0], positions[:,1], positions[:,2])

    plt.show()
