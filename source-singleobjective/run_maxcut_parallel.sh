python3 ./run_parallel_gom.py ./build/GOMEA --approach=0 --problem=3 --alphabet=2 --L=200 --instance=./data/maxcut/maxcut_instance_200_0.txt
mv result/parallel result/maxcut_parallel_GOMEA
python3 ./run_parallel_gom.py ./build/GOMEA --approach=0 --topology=6_-1 --multifos=1 --donorSearch --problem=3 --alphabet=2 --L=200 --instance=./data/maxcut/maxcut_instance_200_0.txt
mv result/parallel result/maxcut_parallel_LKGOMEA_A
python3 ./run_parallel_gom.py ./build/GOMEA --approach=0 --topology=7_-1 --multifos=1 --donorSearch --problem=3 --alphabet=2 --L=200 --instance=./data/maxcut/maxcut_instance_200_0.txt
mv result/parallel result/maxcut_parallel_LKGOMEA_B
