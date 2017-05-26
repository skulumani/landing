#!/bin/bash

NEAR_URL="https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/"
# download the NEAR SPICE data to the current directory
wget --mirror --no-host-directories --cut-dirs=5 --no-verbose --show-progress $NEAR_URL
