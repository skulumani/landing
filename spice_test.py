"""Test the Python SPICE functionality by plotting NEAR

"""

import spiceypy as spice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)

import numpy as np
from urllib import urlretrieve
import os
import pdb

def download_cassini_spice():
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
    if not os.path.isdir('./cassini'):
        os.mkdir('cassini')

    for url in kernel_urls:
        filename = url.split('/')[-1]
        fullfilename = os.path.join('cassini', filename)
        if not os.path.isfile(fullfilename):
            urlretrieve(url, fullfilename)
        spice.furnsh(fullfilename)

def near_state():
    near_id = '-93'
    eros_id = '2000433'
    
    near_body_frame = 'NEAR_SC_BUS_PRIME'
    near_body_frame_id = -93000 
    eros_body_frame = 'IAU_EROS'
    eros_body_frame_id = 2000433

    spice.furnsh('./near_2001.tm')
    step = 1000
    utc = ['Feb 12, 2001 12:00:00', 'Feb 12, 2001 20:05:00']
    etOne = spice.str2et(utc[0])
    etTwo = spice.str2et(utc[1])

    times = np.linspace(etOne, etTwo, step)
    istate = np.zeros((step, 6))
    astate = np.zeros((step, 6))
    ilt = np.zeros_like(times)
    alt = np.zeros_like(times)

    R_sc2int = np.zeros((3, 3, step))
    R_sc2ast = np.zeros((3, 3, step))
    R_ast2int = np.zeros((3, 3, step))

    # angular velocities
    w_sc2int = np.zeros((step, 3))
    w_sc2ast = np.zeros((step, 3))
    w_ast2int = np.zeros((step, 3))

    sc2int_clock = np.zeros(step)
    sc2ast_clock = np.zeros(step)
    ast2int_clock = np.zeros(step)

    for (ii, et) in enumerate(times):
        istate[ii,:], ilt[ii] = spice.spkezr(near_id, et, 'J2000', 'None', eros_id)
        astate[ii,:], alt[ii] = spice.spkezr(near_id, et, 'IAU_EROS', 'None', eros_id)
        
        sclk = spice.sce2c(int(near_id), et)
        # find atttiude states of Eros and NEAR
        try:
            R_sc2int[:, :, ii], w_sc2int[ii, :], sc2int_clock[ii] = spice.ckgpav(near_body_frame_id, sclk, 0.0, 'J2000')
        except:
            pass

        try:
            R_sc2ast[:, :, ii], w_sc2ast[ii, :], sc2ast_clock[ii] = spice.ckgpav(near_body_frame_id,
                    sclk, 0.0, 'IAU_EROS')
        except:
            pass

        try:
            R_ast2int[:, :, ii], w_ast2int[ii, :], ast2int_clock[ii] = spice.ckgpav(eros_body_frame_id, 
                sclk, 0.0, 'J2000')
        except:
            pass
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(istate[:,0],istate[:,1],istate[:,2])

    plt.show()

    spice.kclear()

def near_images():
    """Read NEAR images and test AstroPy
    """
    pass

def astropy_fits():
    """Test reading a fits image
    """
    from astropy.utils.data import download_file
    from astropy.io import fits

    image_file = download_file('http://data.astropy.org/tutorials/FITS-images/HorseHead.fits',
                            cache=True)   

    fits.info(image_file)

    image_data = fits.getdata(image_file, ext=0)

    print(image_data.shape)

    plt.figure()
    plt.imshow(image_data, cmap='gray')
    plt.colorbar()
if __name__=='__main__':
    download_cassini_spice()
    step = 4000
    utc = ['Jun 20, 2004', 'Dec 1, 2005']

    etOne = spice.str2et(utc[0])
    etTwo = spice.str2et(utc[1])
    print("ET One: {}, ET Two: {}".format(etOne, etTwo))

    times = [x*(etTwo-etOne)/step + etOne for x in range(step)]

    positions, lightTimes = spice.spkpos('Cassini', times, 'J2000', 'None', 'SATURN BARYCENTER')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(positions[:,0], positions[:,1], positions[:,2])

    plt.show()
