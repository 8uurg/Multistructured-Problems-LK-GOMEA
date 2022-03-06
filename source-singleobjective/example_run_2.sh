meson compile -C builddir
./builddir/GOMEA --approach=1 --L=100 --timeLimit=3600 --maxEvals=100000 --problem=1_5_5  --alphabet=2 --folder=result_example_2 --seed=42 --hillClimber=1 --donorSearch --scheme=0 --vtr=100
