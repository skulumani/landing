#!/bin/bash

NEAR_URL="https://naif.jpl.nasa.gov/pub/naif/pds/data/near-a-spice-6-v1.0/nearsp_1000/"

set +e
options=("All" "2001" "Quit")
# download the NEAR SPICE data to the current directory
echo "Do you want to download NEAR SPICE data?"
select yn in "${options[@]}"; do
    case $yn in
        All ) eval "wget --mirror --no-host-directories --cut-dirs=5 --no-verbose --show-progress --no-parent -R '*\?C=*' --recursive $NEAR_URL";
            break;;
        2001 ) break;;
        Quit ) exit;;
    esac
done





