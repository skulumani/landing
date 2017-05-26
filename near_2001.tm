KPL/MK

   This meta-kernel lists a subset of kernels from the meta-kernel
   near_v01.tm provided in the data set NEAR-A-SPICE-6-V1.0,
   covering the whole or a part of the customer requested time period
   from 2001-01-01T00:00:00.000 to 2001-02-28T23:59:59.000.

   The documentation describing these kernels can be found in the
   complete data set available at this URL

   ftp://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000

   To use this meta-kernel users may need to modify the value of the
   PATH_VALUES keyword to point to the actual location of the data
   set's ``data'' directory on their system. Replacing ``/'' with ``\''
   and converting line terminators to the format native to the user's
   system may also be required if this meta-kernel is to be used on a
   non-UNIX workstation.

   This meta-kernel was created by the NAIF node's SPICE PDS data set 
   subsetting service version 1.2 on Fri May 26 11:49:54 PDT 2017.

 
   \begindata
 
      PATH_VALUES     = (
                         './data'
                        )
 
      PATH_SYMBOLS    = (
                         'KERNELS'
                        )
 
      KERNELS_TO_LOAD = (
                         '$KERNELS/lsk/naif0007.tls'
                         '$KERNELS/pck/pck00010.tpc'
                         '$KERNELS/pck/erosatt_1998329_2001157_v01.bpc'
                         '$KERNELS/pck/erosatt_1999304_2001151.bpc'
                         '$KERNELS/sclk/near_171.tsc'
                         '$KERNELS/ik/grs12.ti'
                         '$KERNELS/ik/msi15.ti'
                         '$KERNELS/ik/nis14.ti'
                         '$KERNELS/ik/nlr04.ti'
                         '$KERNELS/ik/xrs12.ti'
                         '$KERNELS/fk/eros_fixed.tf'
                         '$KERNELS/spk/de403s.bsp'
                         '$KERNELS/spk/stations.bsp'
                         '$KERNELS/spk/eros80.bsp'
                         '$KERNELS/spk/erosephem_1999004_2002181.bsp'
                         '$KERNELS/spk/math9749.bsp'
                         '$KERNELS/spk/near_erosorbit_nav_v1.bsp'
                         '$KERNELS/spk/near_eroslanded_nav_v1.bsp'
                         '$KERNELS/ck/near_20010101_20010228_v01.bc'
                        )
 
   \begintext
 

