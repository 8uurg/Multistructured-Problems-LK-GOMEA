# meson compile -C builddir
# gdb --args 
# valgrind --track-origins=yes --leak-check=full 
#./build/GOMEA --approach=1 --L=100 --timeLimit=3600 --maxEvals=200000 --problem=1_5_5  --alphabet=2 --folder=result_example_3 --seed=42 --hillClimber=1 --donorSearch --scheme=0 --vtr=100
# ./build/GOMEA --approach=1 --L=125 --timeLimit=3600 --maxEvals=2000000 --problem=3 --instance="./data/maxcut/set2/sg3dl051000.mc" --alphabet=2 --folder=result_example_3 --seed=42 --hillClimber=0 --donorSearch --scheme=0 --vtr=110
# ./build/GOMEA --approach=1 --L=125 --timeLimit=3600 --maxEvals=2000000 --problem=3 --instance="./data/maxcut/set2/sg3dl052000.mc" --alphabet=2 --folder=result_example_3 --seed=42 --hillClimber=0 --donorSearch --scheme=0 --vtr=112
# ./build/GOMEA --approach=1 --noveltyArchive=0.05_100 --L=2744 --timeLimit=3600 --maxEvals=100000000 --problem=3 --instance="./data/maxcut/set2/sg3dl141000.mc" --alphabet=2 --folder=result_example_4 --seed=42 --hillClimber=0 --donorSearch --scheme=0 --vtr=2430
# ./build/GOMEA --approach=0 --L=2744 --timeLimit=3600 --maxEvals=10000000 --problem=3 --instance="./data/maxcut/set2/sg3dl141000.mc" --alphabet=2 --folder=result_example_3 --seed=42 --hillClimber=0 --donorSearch --scheme=0 --vtr=2430
# ./build/GOMEA --approach=0 --L=2744 --timeLimit=3600 --maxEvals=10000000 --problem=3 --instance="./data/maxcut/set2/sg3dl141000.mc" --alphabet=2 --folder=result_example_5 --seed=42 --hillClimber=1 --donorSearch --scheme=0 --vtr=2430
# ./build/GOMEA --approach=0 --L=2744 --timeLimit=3600 --maxEvals=10000000 -C=1 --problem=3 --instance="./data/maxcut/set2/sg3dl141000.mc" --alphabet=2 --folder=result_example_6 --seed=42 --hillClimber=1 --donorSearch --scheme=0 --vtr=2430
./build/GOMEA --approach=1 --noveltyArchive=0.05_100 --L=2744 --timeLimit=3600 --maxEvals=100000000 --problem=3 --instance="./data/maxcut/set2/sg3dl141000.mc" --alphabet=2 --folder=result/gomea --seed=42 --hillClimber=0 --donorSearch --scheme=3 --populationSize=100 --vtr=2430