## Vision based landing

Try to land on an asteroid using a image sensor

## Open CV setup

This will use OpenCV in Python (Anaconda)

* Export the packages in a specific `conda` environment
~~~
$ conda env export -n <env-name> > environment.yml
~~~
* Import to a new environment (on a different computer)
~~~
$ conda env create -f path/to/environment.yml
~~~

### Creating an [explicit](https://conda.io/docs/using/envs.html#share-an-environment) copy of a `conda` environment (not cross platform)

* Export a platform specific package list
~~~
$ conda list --explicit > spec-file.txt
~~~
* Import this file on the same platform (`conda` does not check this! )
~~~
$ conda create --name environment_name --file spec-file.tx


## Data sources

* [NEAR](https://sbn.psi.edu/pds/resource/near/) - this has links to all the NEAR data
    * `near_msi.sh` - will download all the MSI data images over 100GB
    * `near_spice.sh` - will download all the SPICE data
* [Hyabusa](https://sbn.psi.edu/pds/archive/hayabusa.html)
* [SBD Image Viewer](http://sbib.psi.edu/) - sweet tool to look at images from Vesta and Eros
* [NEAR landing imagery](http://sbib.psi.edu/PDS-Eros/Landing/2001-043-MSI/frame/) - directory with NEAR landing phase imagery
* [SPICE Archives](https://naif.jpl.nasa.gov/naif/data_archived.html)

## SPICE Documenation

* [Body ID Codes](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html)
* [SPK Required reading](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html#Integer%20ID%20Codes%20Used%20in%20SPK)
* [Kernel Required reading](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/kernel.html#Loading Kernels)
* [Time](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/time.html)

### MSI Camera

The Multi-Spectral Imager (MSI) on the Near-Earth Asteroid Rendezvous         
(NEAR) spacecraft uses a five-element refractive optical telescope,           
has a field of view of 2.93 x 2.25 degrees, a focal length of 167.35          
mm, and has a spatial resolution of 16.1 x 9.5 m at a range of 100 km.

he camera's optical axis parallel to the
X'-axis of the spacecraft.


The camera specifications are summarized here:

Mass:           Camera           3.7 kg
DPU              4.0 kg
Power:          Camera           1.43 W
DPU              5.49 W
FOV                              2.93 x 2.25 degrees
Spectral Range                   400 - 1100 nm
Refractive Optics                5 Elements
Focal Length                     167.35 mm
Clear Aperture (no cover)        18.6 cm^2
Clear Aperture (with cover)      4.35 cm^2
Frame Size                       537 x 244
Frame Rate                       1 Hz
Frame Size (no compression)      1.6 Mbits
Quantization                     12 bits
Exposure Control                 1 ms to 999 ms
Filter Wheel                     8 Position
Broadband ('Clear')         700 nm
Green                       550 nm
Blue                        450 nm
Red                         760 nm
IR1                         950 nm
IR2                         900 nm                                       
IR3                         1000 nm                                      
IR4                         1050 nm 

A 1 Hz timing signal from the spacecraft synchronizes commands from
the DPU to the camera.  Integration times may be commanded from 1 to
999 ms, effectively varying the sensitivity of the instrument by
nearly three orders of magnitude.  Each full image is made up of 244
x 537 pixels and contains a header of all the parameters associated
with the image, including the time the image was taken, CCD
temperature, exposure time, filter, data compression information,
etc.
