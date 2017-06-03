"""Test the Python SPICE functionality by plotting NEAR

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import spiceypy as spice
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, FuncAnimation

from mpl_toolkits.mplot3d import Axes3D
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)

import cv2
from astropy.utils.data import download_file
from astropy.io import fits

import os
import numpy as np
import pdb

import kernels

def near_state():
    near = kernels.NearKernels
    kernels.getKernels(near)
    metakernel = kernels.writeMetaKernel(near)

    spice.furnsh(metakernel)
    step = 1000
    utc = ['Feb 12, 2001 12:00:00 UTC', 'Feb 12, 2001 20:05:00 UTC']
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
        istate[ii,:], ilt[ii] = spice.spkezr(near.near_id, et, near.inertial_frame, 'None', near.eros_id)
        astate[ii,:], alt[ii] = spice.spkezr(near.near_id, et, near.eros_body_frame, 'None', near.eros_id)
        
        sclk = spice.sce2c(int(near.near_id), et)
        # find atttiude states of Eros and NEAR
        try:
            R_sc2int[:, :, ii], w_sc2int[ii, :], sc2int_clock[ii] = spice.ckgpav(near.near_body_frame_id, sclk, 0.0, near.inertial_frame)
        except:
            pass

        try:
            R_sc2ast[:, :, ii], w_sc2ast[ii, :], sc2ast_clock[ii] = spice.ckgpav(near.near_body_frame_id,
                    sclk, 0.0, near.eros_body_frame)
        except:
            pass

        try:
            R_ast2int[:, :, ii], w_ast2int[ii, :], ast2int_clock[ii] = spice.ckgpav(near.eros_body_frame_id, 
                sclk, 0.0, near.inertial_frame)
        except:
            pass
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(istate[:,0],istate[:,1],istate[:,2])

    plt.show()

    spice.kclear()

def near_image(image_file):
    """Read NEAR images and test AstroPy
    """
    fits.info(image_file, False)
    image_data = fits.getdata(image_file, ext=0)
    print(image_data.shape)

    plt.figure()
    plt.imshow(image_data, cmap='gray')
    plt.axis('off')
    plt.show()

def near_image_flipbook(directory='./images', interval=250):
    """Flip through all the images in a directory
    """
    
    fit_files = [f for f in os.listdir(directory) if f[-3:] == 'fit']
    fit_files = sorted(fit_files)
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.axis('off')

    ims = []
    for (ii, f) in enumerate(fit_files):
        im = ax.imshow(fits.getdata(os.path.join(directory, f), ext=0),
                cmap='gray')
        ims.append([im])

    ani = ArtistAnimation(fig, ims, interval=interval, blit=True,
            repeat_delay=1000)

    return fig, ani

def near_save_gif(gif=False, directory='./images', interval=250):
    """This will output the animation or save
    """
    fig, ani = near_image_flipbook(directory, interval)
    if gif:
        ani.save('landing.gif', dpi=100, writer='imagemagick')
        plt.show()
    else:
        print('fig size: {0} DPI, size in inches {1}'.format(
    fig.get_dpi(), fig.get_size_inches()))
        plt.show()
def astropy_fits():
    """Test reading a fits image
    """

    image_file = download_file('http://data.astropy.org/tutorials/FITS-images/HorseHead.fits',
                            cache=True)   

    fits.info(image_file)

    image_data = fits.getdata(image_file, ext=0)

    print(image_data.shape)

    plt.figure()
    plt.imshow(image_data, cmap='gray')
    plt.colorbar()
    plt.show()

