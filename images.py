"""This module will download imagery for Near
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from astropy.io import fits
import spiceypy as spice

import sys
import os
import urllib.parse
import urllib.request
import time

import pdb
try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Download beautifulsoup4 first")
    sys.exit(0)

cwd = os.path.realpath(os.path.dirname(__file__))
directory = 'images'

if not os.path.isdir(os.path.join(cwd,directory)):
    os.mkdir(os.path.join(cwd,directory))

class NearImages(object):
    """Class that holds all the image data for Near landing
    """

    def __init__(self, year='2001', day='043', downloaded=False):
        """Download all the MSI imagery from a specific day of flight

        This function will download all the MSI imagery from 
        the chosen day. If the day is not valid the response will fail.

        Only downloads the IOFDBL images (Deblurred I/F Images)

        https://sbn.psi.edu/archive/near/NEAR_A_MSI_3_EDR_EROS_ORBIT_V1_0/catalog/msi2erosds.cat

        All the MSI data is available here:

        https://sbn.psi.edu/archive/near/NEAR_A_MSI_3_EDR_EROS_ORBIT_V1_0/
        """
        url = 'https://sbn.psi.edu/archive/near/NEAR_A_MSI_3_EDR_EROS_ORBIT_V1_0/data/' + year + '/' + day + '/' + 'iofdbl/'

        path = os.path.join(cwd, directory)
        self.path = path
        if not downloaded:
            try:
                headers = {"User-Agent": ("Mozilla/5.0 (X11; Linux x86_64)"
                        " AppleWebKit/537.36 (KHTML, like Gecko)"
                        " Chrome/58.0.3029.110 Safari/537.36")}
                with urllib.request.urlopen(url) as response:
                    ii = 0
                    html = response.read() 
                    soup = BeautifulSoup(html, 'lxml')
                    
                    for tag in soup.findAll('a', href=True):
                        href_url = urllib.parse.urljoin(url, tag['href'])
                        ext = os.path.splitext(os.path.basename(href_url))[1]
                        local_filename = os.path.basename(href_url)
                        local_filepath = os.path.join(path, local_filename)
                        if ext == '.fit' or ext == '.lbl':
                            with urllib.request.urlopen(href_url) as current:
                                if not os.path.isfile(local_filepath):                
                                    print("Downloading: {}".format(local_filename))
                                    with open(local_filepath, 'wb') as f:
                                        f.write(current.read())
                                    ii = ii + 1
                                else:
                                    print("Skipping: {}".format(local_filename))
                        
                    print("Download {} files".format(ii))
                    print("All Images from day {0} of {1} should be present".format(
                        day, year))
            except KeyboardInterrupt:
                print("Stopping the download process")
        
        # now store all the filenames into an array
        self.fits = sorted([f for f in os.listdir(path) if f[-3:] == 'fit'])
        self.lbls = sorted([f for f in os.listdir(path) if f[-3:] == 'lbl'])
    
        # useful keys for header
        self.xaxis_key = 'NAXIS1'
        self.yaxis_key = 'NAXIS2'
        self.expms_key = 'NEAR-010'
        self.teltemp_key = 'NEAR-015'
        self.ccdtemp_key = 'NEAR-016'
        self.metstart_key = 'NEAR-017'
        self.metmid_key = 'NEAR-018'
        self.filename_key = 'NEAR-003'
        

        # first element is the scalar. quat_key
        self.quat_key = ['NEAR-019', 'NEAR-020', 'NEAR-021', 'NEAR-022']

        # position vectors from fits file
        self.pos_origin_id = '10' # sun or maybe solar system barycenter 
        self.pos_sun_key = ['NEAR-023', 'NEAR-024', 'NEAR-025']
        self.target_key = ['NEAR-027','NEAR-028', 'NEAR-029']
        self.target_center_key = 'NEAR-030'
        self.target_spice_id_key = 'NEAR-046'
        self.target_name_key = 'NEAR-047' 
        self.slant_dist_key = 'NEAR-057'

    def extract_image_data(self, near, image=''):
        """Extract image data from the Fits
        """
        # preallocate a dictionary/array to store all the information from each
        # image
        spice.furnsh(near.metakernel)
        self.images = []
        # loop over all the images
        for ii, f in enumerate(self.fits): 
            data = fits.open(os.path.join(self.path, f), memmap=True)
            image_data = data[0].data
            header = data[0].header
            data.close()

            # extract the useful information for each image
            image_data = {'image': image_data,
                    'header': header,
                    'path': os.path.join(self.path, f),
                    'filename': header[self.filename_key],
                    'order': ii,
                    'quat': [header[x] for x in self.quat_key],
                    'met_start': header[self.metstart_key],
                    'met_mid': header[self.metmid_key],
                    'et': near.start_et + header[self.metmid_key],
                    'sclk': spice.sce2c(int(near.near_id), near.start_et +
                        header[self.metmid_key]),
                    'utc': spice.timout(near.start_et +
                        header[self.metmid_key],
                        'YYYY MON DD HR:MN:SC.#### (UTC) ::UTC'),
                    'exposure': header[self.expms_key],
                    'telescope_temp': header[self.teltemp_key],
                    'ccd_temp': header[self.ccdtemp_key],
                    'inertial_pos_origin': self.pos_origin_id,
                    'inertial_sun_pos': [header[x] for x in self.pos_sun_key],
                    'inertial_target_pos': [header[x] for x in
                        self.target_key],
                    'target_center_id': header[self.target_spice_id_key],
                    'target_name': header[self.target_name_key],
                    'slant_dist': header[self.slant_dist_key]
                    }

            # compute the spice data at this time stamp using ET, or SCLK
            self.images.append(image_data)
        
        spice.kclear()
#             Ri2b_fits = spice.q2m(quat_fits)
#             # get teh quaternion from teh kernels
#             Ri2b_spice = spice.pxform(self.near.inertial_frame, 
#                     self.near.near_body_frame, et)


