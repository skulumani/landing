#!/bin/bash

NEAR_URL="http://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0"
# download the NEAR SPICE data to the current directory
wget --mirror --no-host-directories --no-verbose --show-progress --wait=10 --random-wait $NEAR_URL
