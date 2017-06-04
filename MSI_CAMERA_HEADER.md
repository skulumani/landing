### MSI Fits image header description

The description of the meta data associated with the MSI imagery is taken 
from [here](https://sbn.psi.edu/archive/near/NEAR_A_MSI_3_EDR_EROS_ORBIT_V1_0/document/instrument/msidefinitions.txt).

Definitions of PDS/NEAR keywords for NEAR MSI parameters                      
========================================================                      
NOTE:  Some of the parameters in the FITS (Flexible Image Transport System)   
       header have no equivalent PDS keyword.  These parameters are placed    
       in parentheses to distinguish them from the PDS keywords.              
                                                                              
                                                                              
NEAR-001 (OFFSET) No PDS Keyword Assigned                                     
-----------------------------------------                                     
For Multi-Spectral Imager (MSI):                                              
                                                                              
  Offset from start of file to image data, in bytes (11520).                  
                                                                              
For Xray/Gamma Ray Spectrometer (XGRS):                                       
                                                                              
  Offset from start of file to table data, in bytes:                          
    XRF => 66240 (Xray Full)                                                  
    XRS => 66240 (Xray Summary)                                               
    GRF => 57600 (Gamma Ray Full)                                             
    GRS => 57600 (Gamma Ray Summary)                                          
    GRB => 11520 (Gamma Ray Burst)                                            
                                                                              
For the Magnetometer (MAG):                                                   
                                                                              
  Offset from start of file to image data, in bytes (17280).                  
                                                                              
For the NEAR Infrared Spectrograph (NIS):                                     
                                                                              
  Offset from start of file to image data, in bytes (18160).                  
                                                                              
For the NEAR Laser Rangefinder (NLR):                                         
                                                                              
  Offset from start of file to image data, in bytes:                          
    LRH => 14400 (High-Rate NEAR Laser Rangefinder)                           
    LRN => 17280 (Normal NEAR Laser Rangefinder)                              
                                                                              
                                                                              
NEAR-002 PRODUCT_VERSION_ID (FILE_FORMAT_VERSION)                             
-------------------------------------------------                             
Version of file format. Refers to the version number of the product listed in 
NEAR-005.                                                                     
                                                                              
The PDS MSI FITS file format is currently '1.01'.  It will be incremented if  
the PDS MSI FITS file format is changed in the future.                        
                                                                              
                                                                              
NEAR-003 SOURCE_FILE_NAME (SDC_FILE_NAME)                                     
-----------------------------------------                                     
Name of the data file at Science Data Center (SDC).  The file naming          
convention is as follows:                                                     
                                                                              
                                                                              
  MnnnnnnnnnnXm_pY.FIT where M is the instrument name MSI, nnnnnnnnnn is the  
  MET stamp on the image (Mission Elapsed Time of image), X is F for full     
  image or S for summary image, m is filter wheel position, p is virtual      
  channel, and Y is P for production or R for Real Time or B for Brass Board  
  Test. Explanations for each component of the name are given below. See      
  NEAR-014 for an explanation of suffixes indicative of calibrated data.      
                                                                              
  nnnnnnnnnn: The MET corresponds to the last spacecraft 1-Hz pulse before the
  end of image integration, i.e., 918 ms before the end of image integration. 
  See Murchie et al. (1998) [MURCHIEETAL1998] for a detailed explanation of   
  relative timing of image integration and the spacecraft pulse.              
                                                                              
  X: A full image is the 537-column, 244-line image returned by MSI. A summary
  image contains 22 lines and 26 columns of 4-bit superpixels, each           
  representing an area 20 pixels wide and 11 pixels high in the original image
  It is used primarily as a vehicle for download of header information.       
                                                                              
  m:  Filter wavelength, nm                                                   
      2 450                                                                   
      1 550                                                                   
      0 700                                                                   
      3 760                                                                   
      5 900                                                                   
      4 950                                                                   
      6 1000                                                                  
      7 1050                                                                  
                                                                              
  p: Virtual Channel (VC) represents the route of image transmission to the   
  ground. VC2 is via the 2 Mbit/sec link directly from MSI to the solid-state 
  recorder, with the data in transfer frame form. VC0 is via the spacecraft   
  1553 bus to the solid state recorder, and VC3 is real time transmission;    
  both are with the data in packet form.                                      
                                                                              
  Y: Real-time processed images are on-ground products created as soon as     
  possible after downlink, before attitude information downlinked separately  
  is associated with the images. Production images have attitude information  
  extracted from spacecraft telemetry and associated with the images.         
  Brassboard images are acquired as parts of onground tests using the         
  spacecraft simulator.                                                       
                                                                              
                                                                              
NEAR-004 (PRODUCT_CREATION_TIME_STAMP) No PDS Keyword Assigned                
--------------------------------------------------------------                
Time the image file was created by SDC.                                       
                                                                              
                                                                              
NEAR-005 SOURCE_PRODUCT_ID (PRODUCT_IDENTIFIER)                               
-----------------------------------------------                               
                                                                              
  MSI.001 is the original FITS file format with a header 8640 bytes long.     
  MSI.002 is the successor format with a header 11520 bytes long. MSI.003 is  
  the next successor format and has a header 11520 bytes long.  MSI.004 is    
  the header for archival PDS files.  The PDS archive will have PDS MSI FITS  
  files with Product ID MSI.004 only with a header 11520 bytes long.          
                                                                              
                                                                              
NEAR-006 SOFTWARE_NAME (PRODUCTION_PROGRAM_NAME)                              
------------------------------------------------                              
Name of the program that produced the data file.                              
                                                                              
    WRITEFITS, for raw data; MSICAL, for 32-bit calibrated data as radiance,  
    of I/F,  OPNAV for 16-bit partially calibrated data, or HL_PDSMSI for PDS 
    archive raw data. See NEAR-014 for an explanation of types of calibrated  
    data. The PDS archive will have PDS MSI FITS files with Program name      
    'HL_PDSMSI' only.                                                         
                                                                              
                                                                              
NEAR-007 SOFTWARE_VERSION_ID (PRODUCTION_PROGRAM_VERSION)                     
---------------------------------------------------------                     
Version number of above program. For calibrated data, version number of the   
program corresponds to calibration version. See NEAR-014 for an explanation of
calibration versions.                                                         
                                                                              
The Version number for the PDS FITS software will contain a tag which         
corresponds to the release tag assigned to the release version of source code.
This tag is assigned using our configuration management software and will be  
of the general form 'v1-11'.  This tag has no significance in processing the  
FITS files. It is used as a means for the NEAR Science Data Center to         
identify revisions of source files used in producing data products.  This     
explanation refers to the FITS header, but the parameter file contains 1.0 to 
indicate this is the first version of the PDS FITS software that is delivered.
Since source code for raw data production will not be part of the archive,    
this tag has no significance.                                                 
                                                                              
                                                                              
NEAR-008 DATA_QUALITY_ID (DATA_QUALITY_FLAG)                                  
--------------------------------------------                                  
Data quality flag.                                                            
                                                                              
DQI definition:                                                               
                                                                              
  c0      = Data Quality Flag [0=good, 1=flagged for quality defects]         
  c1      = Instrument number for FC, CTP and AIU [1 0r 2]                    
  c2 - c6 = File Specific                                                     
  c7      = CCDS Source [0=VC0, 1=VC1, 2=VC2, 3=VC3]                          
                                                                              
                                                                              
NEAR-009 FILTER_NUMBER                                                        
----------------------                                                        
Filter position. The correspondence of filter position with nominal central   
wavelengths is listed under NEAR-003. See Murchie et al. (1998)               
[MURCHIEETAL1998] for detailed information on the bandpasses.                 
                                                                              
                                                                              
NEAR-010 EXPOSURE_DURATION                                                    
--------------------------                                                    
Exposure time in milliseconds. 1-999 for a valid image.  An exposure of 0 ms  
is technically possible but represents an invalid image, containing only      
offsets accumulated during frame transfer and readout. See Murchie et al.     
(1998) [MURCHIEETAL1998] and Hawkins et al. (1997) [HAWKINSETAL1997] for a    
detailed timeline of the processing history of an image by the imager         
electronics.                                                                  
                                                                              
                                                                              
NEAR-011 EXPOSURE_TYPE (AUTOMATIC_EXPOSURE_FLAG)                              
------------------------------------------------                              
Flag for automatic exposure control.                                          
                                                                              
  1 if automatic exposure control was used for this image, and                
  0 if manual exposure was used. In the case of manual exposure, the          
    commanded exposure in ms was executed.                                    
                                                                              
For any sequence that uses the automatic exposure mode, the MSI software      
acquires and processes an additional test image at a commanded time interval  
(NEAR-091), a commanded filter position (NEAR-082), for a commanded exposure  
time (NEAR-088) before the start of the 'science' image sequence. The software
histograms the image, and uses two parameters to determine if the image is    
over- or under-exposed: a target DN level, the the fraction of the image      
allowed to exceed that target. If more than the allowable number of pixels    
exceed the target DN, then the image is considered overexposed and the        
software returns the (commanded) fallback time percentage, multiplied by the  
exposure time used for the test image, as the proper exposure time. Otherwise,
the test image is underexposed and the software scales the commanded text     
exposure time up or down to reach the target DN.                              
                                                                              
To determine the exposure time for images in different filters, the software  
adjusts the proper exposure time using commanded 'relative sensitivity' values
for each filter.                                                              
                                                                              
See Hawkins et al. (1997) [HAWKINSETAL1997] for a more detailed explanation.  
                                                                              
                                                                              
NEAR-012 (ONBOARD_COMPRESSION_ALGORITHM) No PDS Keyword Assigned              
----------------------------------------------------------------              
On-board compression algorithm. Refers to the type of lossless compression, if
any, that was applied to the image prior to downlink.                         
                                                                              
  0 is no compression,                                                        
  1 is Fast compression,                                                      
  2 is Rice compression.                                                      
                                                                              
See Hawkins et al. (1997) [HAWKINSETAL1997] for a detailed explanation of     
these compression algorithms.                                                 
                                                                              
                                                                              
NEAR-013 (PROCESSING_HISTORY_INDICATOR) No PDS Keyword Assigned               
---------------------------------------------------------------               
Calibration indicator (0-raw, 1-radiance, 2-I/F)                              
                                                                              
                                                                              
NEAR-014 (CALIBRATION_FLAG) No PDS Keyword Assigned                           
---------------------------------------------------                           
Calibration version.                                                          
                                                                              
  0    means uncalibrated data,                                               
  1.xx means calibrated data where xx is the version number. Raw image        
       data from the spacecraft are 16-bit unsigned integers. There are three 
       versions of calibrated data, each denoted by a distinct suffix.  *.DSF 
       images are 16-bit signed integer data, in units of DN corrected for    
       dark current, readout smear, and flat-field effects.  *.RAD images are 
       32-bit real numbers in units of radiance (W m^- 2 mcm^-1 sr^-1). *.IOF 
       images are 32-bit real numbers in dimensionless units of I/F           
       (radiance/(pi x irradiance)).                                          
                                                                              
       Calibration version 1.01 utilizes onground flat fields, calibration    
       coefficients based on inflight lunar observations acquired 21 February 
       1996 with the lens cover on, and  (if the lens cover was on) an        
       onground, wavelength-independent attenuation determined from average   
       behavior in all filters with the lens cover on.                        
                                                                              
       Calibration version 1.02 uses flat fields modified based on inflight   
       observations of Earth (23 January 1998) and reanalysis of onground flat
       field measurements, and calibration coefficients based on inflight     
       lunar observations acquired 23 January 1998 with the lens cover off.   
                                                                              
See Murchie et al. (1998) [MURCHIEETAL1998] for a detailed explanation of     
radiometric calibration and the effect of the lens cover.                     
                                                                              
                                                                              
NEAR-015 (TELESCOPE_TEMPERATURE) No PDS Keyword Assigned                      
--------------------------------------------------------                      
Temperature of the telescope body in Celsius. This is calculated from the raw 
value using the equation temp = -100 + 0.803921568627451 x raw_value. The     
telescope body temperature is maintained by heaters near +20C.                
                                                                              
                                                                              
NEAR-016 (CCD_TEMPERATURE) No PDS Keyword Assigned                            
--------------------------------------------------                            
Temperature of CCD in Celsius.  This is measured on the back of the CCD. The  
physical units are calculated from the raw 8- bit value by using the equation 
temp = (-0.5223 x raw_value) + 18.48.                                         
                                                                              
                                                                              
NEAR-017 SPACECRAFT_CLOCK_START_COUNT (IMAGE_ACQUISITION_MET)                 
-------------------------------------------------------------                 
MET of image acquisition in seconds. MET corresponds to the last spacecraft   
1-Hz pulse before the end of image integration, i.e., 918 ms before the end   
of image integration. See Murchie et al. (1998) [MURCHIEETAL1998] for a       
detailed explanation of timing of the 1 Hz pulse relative to image exposure.  
                                                                              
                                                                              
NEAR-018 (MID_EXPOSURE_MET) No PDS Keyword Assigned                           
---------------------------------------------------                           
Image acquisition MET corrected to mid-exposure. This is in decimal units,    
and corrects MET to that at the midpoint of an exposure, the time applicable  
to pointing, using the formula:                                               
                                                                              
MET + 919 ms P (exposure time / 2)                                            
                                                                              
See Murchie et al. (1998) [MURCHIEETAL1998] for a detailed explanation of     
timing of the 1 Hz pulse relative to image exposure.                          
                                                                              
                                                                              
NEAR-019 EMECL_SC_QUATERNION (SPACECRAFT_QUATERNION_Q0)                       
----------------------------------------------------------                    
Spacecraft quaternion in the EME J2000 frame.  REAL, 4 value array.           
                                                                              
  A quaternion is a self-normalized four-vector pointing representation using 
  direction cosine matrices as follows:                                       
                                                                              
         Q = ( q0, q1, q2, q3 )                                               
            = ( cos(T/2), sin(T/2)*u1, sin(T/2)*u2, sin(T/2)*u3 )             
                                                                              
  where  T  is the angle of rotation from the Earth Mean Ecliptic J2000       
            coordinate system centered on the spacecraft to the nominal       
            spacecraft pointing direction; and                                
         u  is the unit vector in the spacecraft pointing direction.          
                                                                              
  Quaternions are used to specify pointing, in this case of a spacecraft, and 
  are used in lieu of other representations, such as Right Ascension,         
  Declination, and Twist.                                                     
                                                                              
                                                                              
NEAR-020 EMECL_SC_QUATERNION (SPACECRAFT_QUATERNION_Q1)                       
---------------------------------------------------------                     
See NEAR-019 for a full explanation                                           
                                                                              
                                                                              
NEAR-021 EMECL_SC_QUATERNION (SPACECRAFT_QUATERNION_Q2)                       
---------------------------------------------------------                     
See NEAR-019 for a full explanation                                           
                                                                              
                                                                              
NEAR-022 EMECL_SC_QUATERNION (SPACECRAFT_QUATERNION_Q3)                       
---------------------------------------------------------                     
See NEAR-019 for a full explanation                                           
                                                                              
                                                                              
NEAR-023 SUN_SC_POSITION_VECTOR (SPACECRAFT_POSITION_VECTOR_X)                
--------------------------------------------------------------                
Spacecraft position in the EME J2000 frame, in km, relative to the origin     
specified in NEAR-026.                                                        
                                                                              
                                                                              
NEAR-024 SUN_SC_POSITION_VECTOR (SPACECRAFT_POSITION_VECTOR_Y)                
--------------------------------------------------------------                
See NEAR-023 for a full explanation.                                          
                                                                              
                                                                              
NEAR-025 SUN_SC_POSITION_VECTOR (SPACECRAFT_POSITION_VECTOR_Z)                
--------------------------------------------------------------                
See NEAR-023 for a full explanation.                                          
                                                                              
                                                                              
NEAR-026 COORDINATE_SYSTEM_CENTER_NAME                                        
--------------------------------------                                        
Reference frame origin for spacecraft, target, and sun positions as listed in 
NEAR-023-025, 027-029, and 031-033 respectively.                              
                                                                              
                                                                              
NEAR-027 TARGET_POSITION_VECTOR (TARGET_POSITION_VECTOR_X)                    
----------------------------------------------------------                    
Target position in the EME J2000 frame, in km, relative to the origin         
specified in NEAR-026.                                                        
                                                                              
                                                                              
NEAR-028 TARGET_POSITION_VECTOR (TARGET_POSITION_VECTOR_Y)                    
----------------------------------------------------------                    
See NEAR-027 for a full explanation.                                          
                                                                              
                                                                              
NEAR-029 TARGET_POSITION_VECTOR (TARGET_POSITION_VECTOR_Z)                    
----------------------------------------------------------                    
See NEAR-027 for a full explanation.                                          
                                                                              
                                                                              
NEAR-030 TARGET_CENTER_DISTANCE                                               
-------------------------------                                               
Range to center of target body from spacecraft in km.                         
                                                                              
                                                                              
NEAR-031 (SUN_POSITION_VECTOR_X) No PDS Keyword Assigned                      
--------------------------------------------------------                      
Sun position in J2000, in km, relative to the origin specified in NEAR-026.   
                                                                              
                                                                              
NEAR-032 (SUN_POSITION_VECTOR_Y) No PDS Keyword Assigned                      
--------------------------------------------------------                      
Sun position in J2000, in km, relative to the origin specified in NEAR-026.   
                                                                              
                                                                              
NEAR-033 (SUN_POSITION_VECTOR_Z) No PDS Keyword Assigned                      
--------------------------------------------------------                      
Sun position in J2000, in km, relative to the origin specified in NEAR-026.   
                                                                              
                                                                              
NEAR-034 (CALIB_COEFF_VERSION) No PDS Keyword Assigned                        
------------------------------------------------------                        
Version of calibration coefficients.                                          
                                                                              
                                                                              
NEAR-035 (0-MS_MET_SUBTRACTED) No PDS Keyword Assigned                        
------------------------------------------------------                        
For calibrated data, the MET of 0-ms exposure subtracted during calibrations. 
                                                                              
                                                                              
NEAR-036 (PSF_FILE_NAME) No PDS Keyword Assigned                              
------------------------------------------------                              
For calibrated data, the point-spread function file name for FFT restoration. 
                                                                              
                                                                              
NEAR-037 (FLAT_FIELD_FILE_NAME) No PDS Keyword Assigned                       
-------------------------------------------------------                       
For calibrated data, the name of the flat field file used to correct the data.
See Murchie et al. (1998) [MURCHIEETAL1998] for a detailed explanation. See   
NEAR-014 for an explanation of different versions of calibrated data.         
                                                                              
                                                                              
NEAR-038 (DARK_CURRENT_FILE_NAME) No PDS Keyword Assigned                     
---------------------------------------------------------                     
For calibrated data, the name of the dark count file used to correct the data.
See Murchie et al. (1998) [MURCHIEETAL1998] for a detailed explanation. See   
NEAR-014 for an explanation of different versions of calibrated data.         
                                                                              
                                                                              
NEAR-039 (CHANNEL_ID) No PDS Keyword Assigned                                 
---------------------------------------------                                 
Transmission channel. 0 is VC0, 2 is VC2, 3 is VC3.  Virtual channel (VC)     
represents the route of image transmission to the ground. VC2 is via the 2    
Mbit/sec link directly from MSI to the solid-state recorder, with the data in 
transfer frame form. VC0 is via the spacecraft 1553 bus to the solid state    
recorder, and VC3 is real time transmission; both VC0 and VC3 are with the    
data in packet form.                                                          
                                                                              
                                                                              
NEAR-040 (IMAGE_OBSERVATION_TYPE) No PDS Keyword Assigned                     
---------------------------------------------------------                     
Image type.  0 is summary image, 1 is full image. A full image is the         
537-column, 244-line image returned by MSI. A summary image contains 22 lines 
and 26 columns of 4-bit superpixels, each representing an area 20 pixels wide 
and 11 pixels high in the original image. It is used primarily as a vehicle   
for download of header information.                                           
                                                                              
                                                                              
NEAR-041 FRAMES                                                               
---------------                                                               
Number of packets (if virtual channel is VC0 or VC3) or transfer frames (if   
virtual channel is VC2) used to downlink the image. See NEAR-003 and NEAR-039 
for information on the virtual channel. An uncompressed image is 184 transfer 
frames.  Use of image compression results in fewer transfer frames being used 
to downlink the image data.                                                   
                                                                              
                                                                              
NEAR-042 SEQUENCE_NUMBER                                                      
------------------------                                                      
Image number in sequence. Images are typically commanded in sequences of up to
eight images. This ordinal number (0-7) refers to the location of the image   
within its sequence. 30 different sequences may be defined at any one time;   
the character of any or all of the different definitions may be modified      
inflight.  These characteristics include manual or automatic exposure; the    
number of images in the sequence; exposure times for each image, which are    
used for manual exposure or are dummy variables for automatic exposure; filter
position for each image; time delay between the images in the sequence; and   
the type of compression used, which applies to all images of the sequence.    
                                                                              
                                                                              
NEAR-043 (SEQ_TABLE_ID) SEQUENCE ID                                           
-----------------------------------                                           
Sequence ID.  This refers to the identifier for a sequence of images (see     
NEAR-042) of which this image is a part. 30 different predefined sequences    
(ID 1 through 30) may be loaded at any one time; the character of any or all  
of the different definitions may evolve over time. Execution of sequence 0    
acquires a single image immediately, using the last commanded settings for    
each of the parameters listed in the explanation to NEAR-042.                 
                                                                              
                                                                              
NEAR-044 (SEQUENCE_NUMBER_OF_IMAGES) No PDS Keyword Assigned                  
------------------------------------------------------------                  
Number of images commanded to be acquired in the sequence identified in       
NEAR-043.  This refers to the predefined sequence of images of which this     
image is a part.  Up to eight images may be included in a sequence.           
                                                                              
                                                                              
NEAR-045 (SEQUENCE_TIME_INTERVAL) No PDS Keyword Assigned                     
---------------------------------------------------------                     
Time interval between images in the sequence identified in NEAR-043, in       
seconds. The time interval can range between 1 and 255 seconds. If the        
commanded time interval is 1, a change in exposure time from the previous     
image causes the actual interval to be 2 seconds. If the commanded time       
interval is 1, an exposure time for the second image of >918 ms will cause    
the actual interval to be 2 seconds. A change in filter position may delay    
the actual image time by up to 2 additional seconds.                          
                                                                              
                                                                              
NEAR-046 (TARGET_SPICE_ID) No PDS Keyword Assigned                            
--------------------------------------------------                            
 Target SPICE ID.  In SPICE system kernel files and subroutines, ephemeris    
 objects, reference frames, and instruments are represented by integer codes. 
                                                                              
 An ephemeris object is any object that may have ephemeris or trajectory data 
 such as a planet, satellite, tracking station, spacecraft, barycenter,       
 asteroid, or comet. Each body in the solar system is associated with a unique
 integer code for use with SPICE.                                             
                                                                              
 The following list contains the SPICE IDs that are currently used for NEAR.  
                                                                              
        SPICE ID   Name                                                       
        --------   ---------------                                            
         0         Solar system barycenter                                    
        399        Earth                                                      
        301        Moon                                                       
        -93        Near Earth Asteroid Rendezvous                             
        1000131    Hyakutake                                                  
        2000253    Mathilde                                                   
        2000433    Eros                                                       
                                                                              
                                                                              
NEAR-047 TARGET_NAME (TARGET)                                                 
-----------------------------                                                 
Target SPICE name.  The name of the TARGET, i.e.  MATHILDE, EROS, etc., uses  
all capital letters.                                                          
                                                                              
                                                                              
NEAR-048 (IMAGE_IDENT) No PDS Keyword Assigned                                
----------------------------------------------                                
Image ID, defined as YYYdddf where YYY is a code for the year (003 is 1996,   
004 is 1997, 005 is 1998, etc.), ddd is the Julian day number of that year,   
and f is the filter position. Times are UT (Universal Time).                  
                                                                              
                                                                              
NEAR-049 (DPU_DECK_TEMPERATURE) No PDS Keyword Assigned                       
-------------------------------------------------------                       
DPU deck temperature in Celsius (scaled msi_tel_msi_dpu_deck)                 
                                                                              
                                                                              
NEAR-050 (CTP_MET) No PDS Keyword Assigned                                    
------------------------------------------                                    
The Command and Telemetry Processor's (CTP) mission elapsed time (MET)        
closest to the MSI MET.  The CTP MET is used to identify the MET used for     
the telescope and deck temperatures.                                          
                                                                              
                                                                              
NEAR-051 (MEAN)                                                               
--------------_                                                               
Average DN level in the image.  Dark space measured at low dark current yields
a DN of about 81 or 87 in odd- or even- numbered columns of raw images (see   
NEAR-014), respectively, due to a bias which is different in odd- or          
even-numbered columns (See Murchie et al. 1998 [MURCHIEETAL1998] for details.)
Missing transfer frames are recorded as zeros. If there are missing transfer  
frames or packets, this variable includes the effects of those zeroes.        
                                                                              
                                                                              
NEAR-052 STANDARD_DEVIATION                                                   
---------------------------                                                   
Standard deviation of DN values in the image. See NEAR-051. Missing frames or 
packets will affect this calculation.                                         
                                                                              
                                                                              
NEAR-053 MAXIMUM                                                              
----------------                                                              
Maximum DN value in the image. See NEAR-051. The maximum possible in raw      
images is 4095.                                                               
                                                                              
                                                                              
NEAR-054 INCIDENCE_ANGLE                                                      
------------------------                                                      
Incidence angle. Not yet implemented for targets other than Eros. Photometric 
angles are measured relative to the surface normal of the plate in the Eros   
shape model (plate model) nearest to the center of the image. Incidence angle 
is between the surface normal at the center of the plate and the vector from  
the center of the plate to the sun, in degrees.                               
                                                                              
                                                                              
NEAR-055 EMISSION_ANGLE                                                       
-----------------------                                                       
Emission angle. Not yet implemented for targets other than Eros. Photometric  
angles are measured relative to the surface normal of the plate in the Eros   
shape model (plate model) nearest to the center of the image. Emission angle  
is between the surface normal at the center of the plate and the vector from  
the center of the plate to the spacecraft, in degrees.                        
                                                                              
                                                                              
NEAR-056 PHASE_ANGLE                                                          
--------------------                                                          
Phase angle. Not yet implemented for targets other than Eros.  Photometric    
angles are measured relative to the surface normal of the plate in the Eros   
shape model (plate model) nearest to the center of the image. Phase angle is  
between the vector to the sun and the vector from to the spacecraft, with the 
vertex located at the intercept of the surface normal of the plate with the   
center of the plate. Units are in degrees.                                    
                                                                              
                                                                              
NEAR-057 SLANT_DISTANCE                                                       
-----------------------                                                       
Spacecraft range to surface in raw image in km.                               
                                                                              
                                                                              
NEAR-058 PIXEL_SATURATION_VALUE                                               
-------------------------------                                               
Pixel saturation value in raw image.                                          
                                                                              
                                                                              
NEAR-059 SATURATED_PIXELS                                                     
-------------------------                                                     
Number of saturated pixels in raw image                                       
                                                                              
                                                                              
NEAR-060 (CAMERA_CURRENT) No PDS Keyword Assigned                             
-------------------------------------------------                             
+28V Camera current, in milliamps.                                            
                                                                              
                                                                              
NEAR-061 (FILTER_SOFTWARE_POSITION) No PDS Keyword Assigned                   
-----------------------------------------------------------                   
Filter wheel software position. This is the position to which the imager      
software has tracked filter wheel position, as determined by the counting     
of motor steps from a previous known position. This value can be reset on     
command, to assist the filter wheel positioning software when one or more     
inoperative fiducials produces an ambiguous position indication.              
                                                                              
                                                                              
NEAR-062 (FW_FIDUCIAL_MASK) No PDS Keyword Assigned                           
---------------------------------------------------                           
Filter wheel fiducial mask. The fiducial position indicators on the filter    
wheel consist of four holes, three large ones to indicate filter wheel        
position and one small one to indicate lock in that position. To read each    
fiducial, an LED emits a light (at a commandable power level) through the     
hole, and that is detectable (at a commandable threshold) by a                
phototransistor. This mask indicates which of the four fiducials marking      
filter wheel position are actually used. 0 means not used, 1 means used.      
See Hawkins et al. (1997) [HAWKINSETAL1997] for a detailed explanation.       
The first of the four bits indicates the fiducial to indicate 'lock in        
position'; the next three bits are the fiducials that indicate filter         
position 0-7. For example, 15 (binary 1111) means all fiducials were used.    
                                                                              
                                                                              
NEAR-063 (DPU_DC/DC_TEMPERATURE) No PDS Keyword Assigned                      
--------------------------------------------------------                      
DPU DC/DC temp (scaled).  Temperature of the DC/DC converter in the instrument
DPU (Digital Processing Unit) in Celsius.  See Hawkins et al. 1997            
[HAWKINSETAL1997] for a detailed engineering description of the DPU.          
                                                                              
                                                                              
NEAR-064 (CAMERA_TEMPERATURE) No PDS Keyword Assigned                         
-----------------------------------------------------                         
Camera temperature (scaled).  Temperature of the telescope body, in Celsius.  
Same as NEAR-015.                                                             
                                                                              
                                                                              
NEAR-065 (FW_POWER_FLAG) No PDS Keyword Assigned                              
------------------------------------------------                              
Filter wheel power flag. This flag is 1 if filter wheel is powered and 0 if it
is off. Normally reads 0, because filter wheel is not moving at the time of   
image acquisition.                                                            
                                                                              
                                                                              
NEAR-066 (FW_ENERGIZE_FLAG) No PDS Keyword Assigned                           
----------------------------------------------------                          
Filter wheel energize flag.  This flag is 1 if filter wheel is energized      
and 0 if it is not.                                                           
                                                                              
                                                                              
NEAR-067 (REC_4_TELEM_CONFIG) No PDS Keyword Assigned                         
-----------------------------------------------------                         
Rec 4 Telemetry Config. Mode of transmission of the type 4 record, i.e.,      
a full 537x244 pixel image. 0 means no full image was produced. 1 is as a     
packetized image (via VC0 and/or VC3), 2 is as transfer frames via VC2, and   
3 is both ways. See NEAR-003 for a description of virtual channels and full   
and summary images.                                                           
                                                                              
                                                                              
NEAR-068 (REC_5_TELEM_CONFIG) No PDS Keyword Assigned                         
-----------------------------------------------------                         
Rec 5 Telemetry Config. Mode of transmission of the type 5 record, i.e., a    
summary 537x244 pixel image. 0 means no summary image was produced. 1 is as a 
packet image (via VC0 and/or VC3). See NEAR-003 for a description of virtual  
channels and full and summary images.                                         
                                                                              
                                                                              
NEAR-069 (LAST_FW_MOVE_COMMAND) No PDS Keyword Assigned                       
-------------------------------------------------------                       
Last filter wheel movement command. The filter can move in two modes: move to 
a filter position, with each filter being located 100 filter-wheel motor steps
apart; or move a commanded number of individual filter-wheel motor steps.     
0 indicates the last command was to go to a filter, and 1 indicates the last  
command was to move a specified number of steps.                              
                                                                              
                                                                              
NEAR-070 (REPEATING_SEQUENCES_FLAG) No PDS Keyword Assigned                   
-----------------------------------------------------------                   
Repeating sequences. 0 is disabled, 1 is enabled. When enabled, a sequence    
repeats continuously either until the imager is powered off or until it is    
cancelled. This capability is intended strictly for onground testing purposes.
                                                                              
                                                                              
NEAR-071 (FW_COMMAND_POSITION) No PDS Keyword Assigned                        
------------------------------------------------------                        
Filter wheel command position. This is the position to which the filter wheel 
was commanded to move, 0-7. See NEAR-003 for identities of these filters.     
                                                                              
                                                                              
NEAR-072 (FW_MOTOR_POWER_LEVEL) No PDS Keyword Assigned                       
-------------------------------------------------------                       
Filter wheel motor power level. This is commandable on a scale of 0 through 7,
with the default being 4.                                                     
                                                                              
                                                                              
NEAR-073 (FW_LED_POWER_LEVEL) No PDS Keyword Assigned                         
-----------------------------------------------------                         
Filter wheel LED power level. This is commandable on a scale of 0-3. See      
NEAR-062 for an explanation of fiducial function.                             
                                                                              
Prior to April 1997 the default level was 1. A large fraction of images       
acquired before that time were not locked in filter position, or were in      
filter wheel fiducial positions indicating a discrepancy from the commanded   
position.  Onground testing confirmed that increasing the LED power level to  
2 greatly improved the consistency between commanded and actual position as   
indicated by the fiducials. The higher power level also increased the         
attainment of lock in position. Subsequently to 10 April 1997 the default     
level was therefore set to 2.                                                 
                                                                              
                                                                              
NEAR-074 (FW_PHOTOTRANSISTOR_THRESHOLD) No PDS Keyword Assigned               
---------------------------------------------------------------               
Filter wheel phototransistor threshold. This is commandable on a scale of 0   
to 7, the default being 4. See NEAR- NEAR-062 for an explanation of fiducial  
function.                                                                     
                                                                              
                                                                              
NEAR-075 (FW_FIDUCIAL) No PDS Keyword Assigned                                
----------------------------------------------                                
Filter wheel fiducial.  The most significant bit of this parameter indicates  
whether the filter wheel is locked.  The remaining three bits indicate filter 
wheel position 0-7. Hence 8 through 15 indicate filter positions 0 through 7, 
respectively, locked in position. 1 through 7 indicate positions 1 through 7  
respectively, not locked in position. 0 is indeterminate, and can mean filter 
position 0 not locked in position, or filter wheel out of position and no     
fiducials visible.                                                            
                                                                              
                                                                              
NEAR-076 (ACTIVE_CTP) No PDS Keyword Assigned                                 
---------------------------------------------                                 
Active CTP at the time of image acquisition. There are two CTPUs on the       
spacecraft. 0 is CTP#1, 1 is CTP#2. (CTP is Command and Telemetry Processor)  
                                                                              
                                                                              
NEAR-077 (MAXIMUM_PIXEL_VALUE_DIAGNOSTIC) No PDS Keyword Assigned             
-----------------------------------------------------------------             
Maximum pixel value diagnostic.  During any second in which the MSI is not    
otherwise exposing or reading out an image, it reads the image from the camera
and computes, as a camera diagnostic, the maximum in a sampling grid of every 
16th pixel in every 16th row. This value is not updated while other images are
being taken.                                                                  
                                                                              
                                                                              
NEAR-078 (EXPOSURE_TIME_MODE_COMMAND_ECHO) No PDS Keyword Assigned            
------------------------------------------------------------------            
Exposure time mode command echo. An echo of the commanded exposure mode, 0 for
manual and 1 for automatic.  See NEAR-011 for an explanation of automatic     
exposure.                                                                     
                                                                              
                                                                              
NEAR-079 (EXPOSURE_TIME_PAR_COMMAND_ECHO) No PDS Keyword Assigned             
-----------------------------------------------------------------             
Exposure time parameter command echo. An echo of the commanded exposure time, 
as given for images in this sequence in NEAR-104 to NEAR-111. See these and   
NEAR-011 for an explanation of the usage.                                     
                                                                              
                                                                              
NEAR-080 (AIU_RT-RT_DATA_CONTROL) No PDS Keyword Assigned                     
---------------------------------------------------------                     
AIU RT-RT Data Control. Starting value for onground, incrementing test        
parameter. Values 0-255. AIU is Attitude Interface Unit.                      
                                                                              
                                                                              
NEAR-081 (CCD_TEST_PORT_MODE) No PDS Keyword Assigned                         
-----------------------------------------------------                         
CCD test port mode. 0 is normal, 1 indicates image sent to test port (see     
Hawkins et al. 1997 [HAWKINSETAL1997] for a detailed explanation of the use of
the test port). The latter setting is used for onground testing. The DPU      
software sets the mode to normal after each image acquisition.                
                                                                              
                                                                              
NEAR-082 (AUTOEXPOSURE_TEST_FILTER) No PDS Keyword Assigned                   
-----------------------------------------------------------                   
Autoexposure test filter.  If an autoexposure test image was acquired prior to
acquisition of the sequence of which this image is a part, i.e. if NEAR-011 is
1, this is the filter position commanded to be used in that test. See NEAR-011
for an explanation of automatic exposure.                                     
                                                                              
                                                                              
NEAR-083 (PIXELS_PER_COMPRESSION_BLOCK) No PDS Keyword Assigned               
---------------------------------------------------------------               
Number of pixels per compression block for a compressed image. 0 means 16     
pixels/block, 1-15 gives the number of pixels per block. See Hawkins et al.   
(1997) [HAWKINSETAL1997] and NEAR-012 for discussions of on-board compression 
algorithms.                                                                   
                                                                              
                                                                              
NEAR-084 RIGHT_ASCENSION                                                      
------------------------                                                      
The right ascension element provides the right ascension value, in EMEJ2000   
coordinates, of the center pixel of the image. Right ascension is defined as  
the arc of the celestial equator between the vernal equinox (as of            
2000-01-01T12:00:00) and the point where the hour circle through the given    
body intersects the Earth's mean equator (reckoned eastward). See declination.
                                                                              
                                                                              
NEAR-085 DECLINATION                                                          
--------------------                                                          
The declination element provides the declination, in EMEJ2000 coordinates,    
of the center pixel of the image.  Declination is the angle measured between  
the boresight vector, which corresponds to the center pixel, and the plane    
containing the celestial equator.  Declination corresponds to the latitude    
and is positive when measured north of the celestial equator and negative     
when measured south of the celestial equator.  See right ascension.           
                                                                              
                                                                              
NEAR-086 CELESTIAL_NORTH_CLOCK_ANGLE                                          
------------------------------------                                          
The celestial clock north angle element specifies the direction of            
celestial north at the center of ana image.  It is measured from the          
'upward' direction, clockwise to the direction tward celestial north          
(declination = +9- degrees), when the image is displayed as defined by        
the SAMPLE_DISPLAY_DIRECTION and LINE_DISPLAY-DIRECTION elements.  The        
epoch of the celestial coordinate system is J2000 unless otherwise indicated. 
Note: This element bears a simple relationship to the value of TWIST_ANGLE.   
                                                                              
When TWIST_ANGLE_TYPE = DEFAULT,                                              
CELESTIAL_NORTH_CLOCK_ANGLE = (180 - TWIST_ANGLE) mod 369;                    
when TWIST_ANGLE_TYPE = GALILEO,                                              
CELESTIAL_NORTH_CLOCK_ANGEL = (270 - TWIST_ANGLE) mod 360.                    
                                                                              
Note: For images pointed near either pole, the value varies significantly     
across the image; in these cases, the element is very sensitive to the        
accuracy of the pointing information.                                         
                                                                              
                                                                              
NEAR-087 (IMAGER_CURRENT_LIMIT) No PDS Keyword Assigned                       
-------------------------------------------------------                       
Imager current limit. Unscaled 8-bit number. If the imager current exceeds    
this value for 3 consecutive seconds, the instrument will shut off.  Use the  
equation '500 - 4.03 * ImagerCurrentLimit' to convert this number to mA       
                                                                              
                                                                              
NEAR-088 (AUTOEXPOSURE_TEST_EXPOSURE_TIME) No PDS Keyword Assigned            
------------------------------------------------------------------            
Autoexposure test image exposure time.  If an autoexposure test image was used
for the sequence of which this image is a part, i.e. if NEAR-011 is 1, this is
the exposure time, in milliseconds, of that test image. See NEAR-011 for an   
explanation of automatic exposure.                                            
                                                                              
                                                                              
NEAR-089 (FW_MOTOR_STEP_TIME) No PDS Keyword Assigned                         
-----------------------------------------------------                         
Filter wheel motor step time.  This is the commanded time in milliseconds     
between filter wheel motor steps, when moving in either mode. See NEAR-069.   
                                                                              
                                                                              
NEAR-090 (FW_COMMANDED_STEPS) No PDS Keyword Assigned                         
-----------------------------------------------------                         
Filter wheel commanded number of steps. The last number of steps commanded    
when filter wheel motion was commanded in motor steps instead of filter wheel 
position. To be used in case of failure of the filter wheel. See NEAR-069.    
                                                                              
                                                                              
NEAR-091 (AUTOEXPOSE_TEST_ADV_TIME) No PDS Keyword Assigned                   
-----------------------------------------------------------                   
Autoexpose test image advance time.  If an autoexposure test was used for this
image, i.e. if NEAR-011 is 1, this is the delay time in seconds after the     
autoexposure test image until the first science image of the sequence is      
taken.  See NEAR-011 for an explanation of automatic exposure.                
                                                                              
                                                                              
NEAR-092 (MOST_RECENT_SEQUENCE) No PDS Keyword Assigned                       
-------------------------------------------------------                       
Most recent image acquisition sequence. ID of last sequence executed. See     
NEAR-042.                                                                     
                                                                              
                                                                              
NEAR-093 (SEQUENCE_SCHEDULED_TIME) No PDS Keyword Assigned                    
----------------------------------------------------------                    
Sequence scheduled time. See NEAR-042 for the definition of an image sequence.
This is the MET of the scheduled start of the sequence of which this image    
is a part.                                                                    
                                                                              
                                                                              
NEAR-094 SEQUENCE_NUMBER (Image Number In Sequence)                           
---------------------------------------------------                           
Same as NEAR-042                                                              
                                                                              
                                                                              
NEAR-095 SEQUENCE_TABLE_ID                                                    
--------------------------                                                    
Same as NEAR-043                                                              
                                                                              
                                                                              
NEAR-096 (NUMBER_OF_IMAGES) No PDS Keyword Assigned                           
---------------------------------------------------                           
Number of images to acquire.  This refers to the number of images in the      
predefined sequence of which this image is a part.  Up to eight images may be 
included in a sequence. See NEAR-044.                                         
                                                                              
                                                                              
NEAR-097 EXPOSURE_TYPE                                                        
----------------------                                                        
Redundant to NEAR-011.                                                        
                                                                              
                                                                              
NEAR-098 (SEQUENCE_TIME_INTERVAL) No PDS Keyword Assigned                     
---------------------------------                                             
Redundant to NEAR-045.                                                        
                                                                              
                                                                              
NEAR-099 (REPEATING_SEQUENCES_FLAG) No PDS Keyword Assigned                   
---------------------                                                         
Redundant to NEAR-070.                                                        
                                                                              
                                                                              
NEAR-100 (PIXELS_PER_COMPRESSION_BLOCK)                                       
--------------------------------------                                        
Pixels per compression block. Redundant to NEAR-083.                          
                                                                              
                                                                              
NEAR-101 (ONBOARD_COMPRESSION_ALGORITHM) No PDS Keyword Assigned              
----------------------------                                                  
Redundant to NEAR-012.                                                        
                                                                              
                                                                              
NEAR-102 (DPCM_FLAG) No PDS Keyword Assigned                                  
--------------------------------------------                                  
DPCM (Digital Pulse Code Modulation) flag.  0 is off, 1 is on. Used in        
conjunction with Fast or Rice lossless compression to conserve bits. See      
Hawkins et al. (1997) [HAWKINSETAL1997] for a detailed explanation.           
                                                                              
                                                                              
NEAR-103 INST_CMPRS_QUANTZ_TBL_ID (LOSSY_COMPRESSION_LOOKUP_TABLE)            
------------------------------------------------------------------            
Lossy lookup table. Refers to the seven lookup tables used optionally to      
translate 12-bit DN levels into 8-bit DN levels. Value is NONE if 12-bit data 
format retained. For conversion to 8 bits, values range from TABLE1 to TABLE7.
Each performs the conversion proportionally to a different function, such as  
logarithmic or linear. 8-bit DNs are reconverted to 12-bit values during      
ground decompression. See Hawkins et al. (1997) [HAWKINSETAL1997] for a       
detailed explanation. Conversions are listed in the file lossy.txt, and shown 
graphically in the files lossy.gif, lossy.jpg, lossy.png, lossy.tif in        
/DOCUMENT/INSTRUMENT                                                          
                                                                              
                                                                              
NEAR-104-111 (IMAGE_N_EXPOSURE_TIME) No PDS Keyword Assigned                  
------------------------------------------------------------                  
This definition applies to the eight NEAR parameters with N from 0 to 7.      
                                                                              
Image N exposure time, in milliseconds, for N in the range 0-7, for up to     
eight images in the predefined sequence of which this image is a part. These  
are used for manual exposure or are dummy variables for automatic exposure.   
See NEAR-011.                                                                 
                                                                              
                                                                              
NEAR-112-119 (IMAGE_N_FW_POSITION) No PDS Keyword Assigned                    
----------------------------------------------------------                    
This definition applies to the eight NEAR parameters with N from 0 to 7.      
                                                                              
Image N filter wheel position, for N in the range 0-7 for up to eight images  
in the predefined sequence of which this image is a part.  See NEAR-003       
for identities of these filters.                                              
                                                                              
                                                                              
NEAR-120 VERTICAL_PIXEL_SCALE                                                 
-----------------------------                                                 
The vertical pixel scale is the spatial resolution of the center pixel in the 
vertical direction, measured at the surface of EROS.  This value is computed  
at a distance equal to the range to the surface at the instant the image      
was acquired, computed as: 2 * (range to surface) * tan (half of the angular  
width of the field of view in the vertical direction).  Refer to the most     
recent MSI instrument kernel for the width of the field of view in the        
vertical direction.  Vertical pixel scale is represented in km/pixel at       
the center pixel.  Because of the irregular shape of EROS, this value is      
only valid at the center pixel.                                               
                                                                              
                                                                              
NEAR-121 HORIZONTAL_PIXEL_SCALE                                               
-------------------------------                                               
The horizontal pixel scale is the spatial resolution of the center pixel in   
the horizontal direction, measured at the surface of EROS.  This value is     
computed at a distance equal to the range to the surface at the instant the   
image was acquired, computed as: 2 * (range to surface) * tan (half of the    
angular width of the field of view in the horizontal direction).  Refer to the
most recent MSI instrument kernel for the width of the field of view in the   
horizontal direction.  Horizontal pixel scale is represented in km/pixel at   
the center pixel.  Because of the irregular shape of EROS, this value is      
only valid at the center pixel.                                               
