"""This module will download imagery for Near
"""
import sys
import os
import urlparse
import urllib2
import time

import pdb
try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Download beautifulsoup4 first")
    sys.exit(0)

cwd = os.path.realpath(os.path.dirname(__file__))
directory = 'images'

if not os.path.isdir(directory):
    os.mkdir(directory)

def getNearMSI(year='2001', day='043'):
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
    try:
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"}
        ii = 0
        request = urllib2.Request(url, None, headers)
        html = urllib2.urlopen(request)
        soup = BeautifulSoup(html.read(), 'lxml')
        
        for tag in soup.findAll('a', href=True):
            href_url = urlparse.urljoin(url, tag['href'])
            ext = os.path.splitext(os.path.basename(href_url))[1]
            if ext == '.fit' or ext == '.lbl':
                current = urllib2.urlopen(href_url)
                local_filename = os.path.basename(href_url)
                local_filepath = os.path.join(path, local_filename)

                if not os.path.isfile(local_filepath):                
                    print("Downloading: {}".format(local_filename))
                    f = open(local_filepath, 'wb')
                    f.write(current.read())
                    f.close()
                    ii = ii + 1
                else:
                    print("Skipping: {}".format(local_filename))

        print("Download {} files".format(ii))
        return 0
    except KeyboardInterrupt:
        print "Exiting"
        return 1

