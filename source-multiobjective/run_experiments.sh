#!/bin/bash
set -euo pipefail

# Perform compilation (eg. make sure we are always using the latest version)
# - Setting up the build directory
#   Note: this is allowed to fail (if there is already a directory set up) 
meson build --buildtype=debugoptimized || true
# - Perform the actual compilation
meson compile -C build
# Run commands listed in experiments.txt
cat ./experiments.txt | xargs -P 0 -n 1 -d '\n' sh -c
# Tar and compress /results into results.tar.gz
# In order for this to work useful ensure that all results are in ./results
tar -czf ./results.tar.gz ./results