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

class NearKernels(object):
    """List of urls and kernels for the Near mission

	This only downloads data for 2001, not the whole mission.

    More data on Near is available:
    https://pdssbn.astro.umd.edu/data_sb/missions/near/index.shtml
    """
    near_id = '-93'
    eros_id = '2000433'
    
    near_body_frame = 'NEAR_SC_BUS_PRIME'
    near_body_frame_id = -93000 
    eros_body_frame = 'IAU_EROS'
    eros_body_frame_id = 2000433

    inertial_frame = 'J2000'
    
    Lsk_url = 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/a_old_versions/naif0011.tls'
    Ck_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/ck/near_20010101_20010228_v01.bc'
    Sclk_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/sclk/near_171.tsc'

    PckEros1_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/pck/erosatt_1998329_2001157_v01.bpc'
    PckEros2_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/pck/erosatt_1999304_2001151.bpc'
    Pck_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/pck/pck00010.tpc'

    Fk_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/fk/eros_fixed.tf'

    Ikgrs_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/ik/grs12.ti'
    Ikmsi_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/ik/msi15.ti'
    Iknis_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/ik/nis14.ti'
    Iknlr_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/ik/nlr04.ti'
    Ikxrs_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/ik/xrs12.ti'
   
    SpkPlanet_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/spk/de403s.bsp'
    SpkEros_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/spk/eros80.bsp'
    SpkEros2_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/spk/erosephem_1999004_2002181.bsp'
    SpkMath_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/spk/math9749.bsp'
    SpkNearLanded_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/spk/near_eroslanded_nav_v1.bsp'
    SpkNearOrbit_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/spk/near_erosorbit_nav_v1.bsp'
    SpkStations_url = 'https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/data/spk/stations.bsp'

    Lsk = getPathfromUrl(Lsk_url)
    Ck = getPathfromUrl(Ck_url)
    Sclk = getPathfromUrl(Sclk_url)

    PckEros1 = getPathfromUrl(PckEros1_url)
    PckEros2 = getPathfromUrl(PckEros2_url) 
    Pck = getPathfromUrl(Pck_url) 

    Fk = getPathfromUrl(Fk_url) 

    Ikgrs = getPathfromUrl(Ikgrs_url)
    Ikmsi = getPathfromUrl(Ikmsi_url)
    Iknis = getPathfromUrl(Iknis_url)
    Iknlr = getPathfromUrl(Iknlr_url)
    Ikxrs = getPathfromUrl(Ikxrs_url)
   
    SpkPlanet = getPathfromUrl(SpkPlanet_url)
    SpkEros = getPathfromUrl(SpkEros_url)
    SpkEros2 = getPathfromUrl(SpkEros2_url)
    SpkMath = getPathfromUrl(SpkMath_url)
    SpkNearLanded = getPathfromUrl(SpkNearLanded_url)
    SpkNearOrbit = getPathfromUrl(SpkNearOrbit_url)
    SpkStations = getPathfromUrl(SpkStations_url)
    
    urlList = [Lsk_url, Ck_url, Sclk_url, Pck_url, PckEros1_url, PckEros2_url, Fk_url,
               Ikgrs_url, Ikmsi_url, Iknis_url, Iknlr_url, Ikxrs_url,
               SpkPlanet_url, SpkEros_url, SpkEros2_url, SpkMath_url,
               SpkNearLanded_url, SpkNearOrbit_url, SpkStations_url]

    kernelList = [Lsk, Ck, Sclk, Pck, PckEros1, PckEros2, Fk,
               Ikgrs, Ikmsi, Iknis, Iknlr, Ikxrs,
               SpkPlanet, SpkEros, SpkEros2, SpkMath,
               SpkNearLanded, SpkNearOrbit, SpkStations]

    nameList = [getKernelNameFromUrl(url) for url in urlList]

    kernelDescription = 'Metal Kernel for 2001 NEAR orbit and landing'
    
class CassiniKernels(object):
    """List of urls and kernels for the Cassini mission

    More data on Cassini is available:
    https://naif.jpl.nasa.gov/pub/naif/CASSINI/
    """
    Lsk_url = 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/a_old_versions/naif0011.tls'
    Sclk_url = 'https://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/sclk/cas00171.tsc'
    Pck_url = 'https://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/pck/cpck09May2017.tpc'
    Fk_url = 'https://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/fk/release.11/cas_v40.tf'
    Ck_url = 'https://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/ck/04135_04171pc_psiv2.bc'
    Spk_url = 'https://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/spk/981005_PLTEPH-DE405S.bsp'
    Ik_url = 'https://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/ik/release.11/cas_iss_v10.ti'
    TourSpk_url = 'https://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/spk/030201AP_SK_SM546_T45.bsp'
    satSpk_url = 'https://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/spk/020514_SE_SAT105.bsp'
    
    Lsk = getPathfromUrl(Lsk_url)
    Sclk = getPathfromUrl(Sclk_url)
    Pck = getPathfromUrl(Pck_url)
    Fk = getPathfromUrl(Fk_url)
    Ck = getPathfromUrl(Ck_url)
    Spk = getPathfromUrl(Spk_url)
    Ik = getPathfromUrl(Ik_url)
    TourSpk = getPathfromUrl(TourSpk_url)
    satSpk = getPathfromUrl(satSpk_url)
    
    urlList = [Lsk_url, Sclk_url, Pck_url, Fk_url, 
               Ck_url, Spk_url, Ik_url, TourSpk_url,
               satSpk_url]
    kernelList = [Lsk, Sclk, Pck, Fk, Ck,
                Spk, Ik, TourSpk, satSpk]
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
    
