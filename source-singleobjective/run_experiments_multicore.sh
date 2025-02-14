#!/bin/bash
set -euo pipefail
#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
#
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
#
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

# Note: This file is the same as run_experiments.sh
# just the number of parallel runs has been reduced to
# account for multicore experiments.

# Perform compilation (eg. make sure we are always using the latest version)
# - Setting up the build directory
#   Note: this is allowed to fail (if there is already a directory set up) 
meson build --buildtype=debugoptimized || true
# - Perform the actual compilation
meson compile -C build
# Run commands listed in experiments.txt
cat ./experiments.txt | xargs -P 3 -n 1 -d '\n' sh -c
# Tar and compress /results into results.tar.gz
# In order for this to work useful ensure that all results are in ./results
tar -czf ./results.tar.gz ./results