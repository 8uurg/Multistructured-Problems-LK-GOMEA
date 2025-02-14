# GOMEA

## Building

### Meson
```
meson builddir --buildtype=release
meson compile -C builddir
```

Build results will be in `builddir`.

### Makefile
The original build process is invoked through make:
```
make
``` 

**Notes:**
- Only works on Linux.
- Requires you to configure the path to the Python include directory manually.

## Example: Running
Assuming meson was used to build, replace `./build/` with `./` otherwise.
```
./build/GOMEA --approach=1 --scheme=0 --problem=3 --alphabet=2 --L=100 --instance=./data/maxcut/maxcut_instance_100_0.txt
```

# Credit
 DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
 
This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.

Project leaders: Peter A.N. Bosman, Tanja Alderliesten
Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
Main code developer: Arthur Guijt