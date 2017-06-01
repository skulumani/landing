"""This module will download kernels and setup SPICE
"""
import os
import time
from urllib import urlretrieve

cwd = os.path.realpath(os.path.dirname(__file__))
directory = 'kernels'

if not os.path.isdir(directory):
    os.mkdir(directory)

def getKernelNameFromUrl(url):
    """Extract the Kernal name from a URL
    """
    return url.split('/')[-1]

def getPathfromUrl(url):
    """Extract the path from the url
    """
    return os.path.join(cwd, directory, getKernelNameFromUrl(url))

def cleanupFile(path):
    """Delete a file from the given path
    """

class CassiniKernels(object):
    """List of urls and kernels for the Cassini mission

    More data on Cassini is available:
    https://naif.jpl.nasa.gov/pub/naif/CASSINI/
    """
    cassLsk_url = 'http://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/a_old_versions/naif0011.tls'
    cassSclk_url = 'http://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/sclk/cas00171.tsc'
    cassPck_url = 'http://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/pck/cpck09May2017.tpc'
    cassFk_url = 'http://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/fk/release.11/cas_v40.tf'
    cassCk_url = 'http://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/ck/04135_04171pc_psiv2.bc'
    cassSpk_url = 'http://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/spk/981005_PLTEPH-DE405S.bsp'
    cassIk_url = 'http://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/ik/release.11/cas_iss_v10.ti'
    cassTourSpk_url = 'http://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/spk/030201AP_SK_SM546_T45.bsp'
    satSpk_url = 'http://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/spk/020514_SE_SAT105.bsp'
    
    cassLsk = getPathfromUrl(cassLsk_url)
    cassSclk = getPathfromUrl(cassSclk_url)
    cassPck = getPathfromUrl(cassPck_url)
    cassFk = getPathfromUrl(cassFk_url)
    cassCk = getPathfromUrl(cassCk_url)
    cassSpk = getPathfromUrl(cassSpk_url)
    cassIk = getPathfromUrl(cassIk_url)
    cassTourSpk = getPathfromUrl(cassTourSpk_url)
    satSpk = getPathfromUrl(satSpk_url)
    
    urlList = [cassLsk_url, cassSclk_url, cassPck_url, cassFk_url, 
               cassCk_url, cassSpk_url, cassIk_url, cassTourSpk_url,
               satSpk_url]
    kernelList = [cassLsk, cassSclk, cassPck, cassFk, cassCk,
                cassSpk, cassIk, cassTourSpk, satSpk]
    nameList = [getKernelNameFromUrl(url) for url in urlList]

    kernelDescription = 'Metal Kernel for Cassini Orbiter'

def cleanupKernels(kernelObj=CassiniKernels):
    """Delete all the Kernels
    """
    for kernel in kernelObj.kernelList:
        path = os.path.join(cwd, directory, kernel)
        if os.path.exists(path):
            os.remove(path)
            return 0
        else:
            print("Path doesn't exist")
            return 1


def attemptDownload(url, kernelName, targetFileName, num_attempts=5):
    """Download the file from a specific url
    """
    current_attempt = 0
    while current_attempt < num_attempts:
        try:
            print("Attempting to download kernel: {}".format(kernelName))
            urlretrieve(url, targetFileName)
            break
        except:
            pass
        current_attempt = current_attempt + 1
        print("Attempting to download kernel again...")
        time.sleep(2 + current_attempt)

    if current_attempt >= num_attempts:
        raise BadException("Error downloading kernel: {}. Check if it exists at url: {}".format(kernelName, url))


def getKernels(kernelObj=CassiniKernels):
    """Download all the Kernels
    """
        
    for url in kernelObj.urlList:
        kernelName = getKernelNameFromUrl(url)
        kernelFile = os.path.join(cwd, directory, kernelName)

        if not os.path.isfile(kernelFile):
            attemptDownload(url, kernelName, kernelFile, 5)

    return 0

def writeMetaKernel(kernelObj, filename='testKernel.tm'):
    """Write a user defined meta kernel file
    """
    with open(os.path.join(cwd, directory, filename), 'w') as metaKernel:
        metaKernel.write('\\begintext\n\n')
        metaKernel.write('Created: Shankar Kulumani\n')
        if kernelObj.kernelDescription:
            metaKernel.write('Description: {}\n'.format(kernelObj.kernelDescription))

        metaKernel.write('\n')

        metaKernel.write('\\begindata\n\n')
        metaKernel.write('PATH_VALUES = (\n')
        metaKernel.write('\'{0}\'\n'.format(os.path.join(cwd, directory)))
        metaKernel.write(')\n\n')

        metaKernel.write('PATH_SYMBOLS = (\n')
        metaKernel.write('\'KERNELS\'\n')
        metaKernel.write(')\n\n')

        metaKernel.write('KERNELS_TO_LOAD = (\n')
        for kernel in kernelObj.nameList:
            metaKernel.write('\'$KERNELS/{0}\'\n'.format(kernel))

        metaKernel.write(')\n')
        metaKernel.close()
    print("Finished writing metakernel.")

    return os.path.join(cwd, directory, filename)

if __name__ == '__main__':
    cass = CassiniKernels
    getKernels(cass)
    writeMetaKernel(cass, 'cassini.tm')
    
