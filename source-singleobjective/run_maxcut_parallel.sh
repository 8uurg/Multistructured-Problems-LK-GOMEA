#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
#
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
#
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

python3 ./run_parallel_gom.py ./build/GOMEA --approach=0 --problem=3 --alphabet=2 --L=200 --instance=./data/maxcut/maxcut_instance_200_0.txt
mv result/parallel result/maxcut_parallel_GOMEA
python3 ./run_parallel_gom.py ./build/GOMEA --approach=0 --topology=6_-1 --multifos=1 --donorSearch --problem=3 --alphabet=2 --L=200 --instance=./data/maxcut/maxcut_instance_200_0.txt
mv result/parallel result/maxcut_parallel_LKGOMEA_A
python3 ./run_parallel_gom.py ./build/GOMEA --approach=0 --topology=7_-1 --multifos=1 --donorSearch --problem=3 --alphabet=2 --L=200 --instance=./data/maxcut/maxcut_instance_200_0.txt
mv result/parallel result/maxcut_parallel_LKGOMEA_B
