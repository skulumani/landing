## MSI Camera

### Reference frame

The MSI boresight is aligned with teh +X vector of the NEAR_MSI frame.

Using the Euler angles given in this kernel, the components of
the NEAR_MSI frame relative to the NEAR_SC_BUS_PRIME frame
are found by the SPICE subroutine SXFORM to be as follows:

X:      .9999988429    -.0004838414     .0014422523
Y:      .0004838419     .9999998829     .0000000000
Z:     -.0014422522     .0000006978     .9999989600

This indicates that the NEAR_MSI boresight points towards
(.9999988429, -.0004838414, .0014422523) as measured in the
NEAR_SC_BUS_PRIME frame.

This is the rotation matrix which transorms vectors in the NEAR_SC_BUS_PRIME 
frame to the NEAR_MSI frame.

### Camera details
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
