#!/bin/bash
for num_parameters in {6,12,25,50,100}
do
	if [ $num_parameters -eq 6 ]; then
		limit=10000
		interval=1000
	fi
	if [ $num_parameters -eq 12 ]; then
		limit=50000
		interval=5000
	fi
	if [ $num_parameters -eq 25 ]; then
		limit=500000
		interval=50000
	fi
	if [ $num_parameters -eq 50 ]; then
		limit=5000000
		interval=500000
	fi
	if [ $num_parameters -eq 100 ]; then
		limit=20000000
		interval=2000000
	fi
	for i in {1..100}
	do
		# Standard version, no mutation, no stopping small population
		./MO_GOMEA 4 2 $num_parameters 1000 $limit $interval
		# With weak mutation
		#./MO_GOMEA -m 4 2 $num_parameters 1000 $limit $interval
		# With strong mutation
		#./MO_GOMEA -M 4 2 $num_parameters 1000 $limit $interval
		# With stopping small populations
		#./MO_GOMEA -z 4 2 $num_parameters 1000 $limit $interval
		for f in elitist*.dat;
			do mv $f ${num_parameters}_${i}_${f};
		done
	done
done

