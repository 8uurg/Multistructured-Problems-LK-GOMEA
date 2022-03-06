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
