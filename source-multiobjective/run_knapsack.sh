#!/bin/bash
#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
#
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
#
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

for num_parameters in {100,250,500,750}
do
	if [ $num_parameters -eq 100 ]; then
		limit=5000000
		interval=500000
	fi
	if [ $num_parameters -eq 250 ]; then
		limit=10000000
		interval=1000000
	fi
	if [ $num_parameters -eq 500 ]; then
		limit=15000000
		interval=1500000
	fi
	if [ $num_parameters -eq 750 ]; then
		limit=20000000
		interval=2000000
	fi
	for i in {1..100}
	do
		# Standard version, no mutation, no stopping small population
		./MO_GOMEA -r 2 2 $num_parameters 2000 $limit $interval
		# With weak mutation
		#./MO_GOMEA -r -m 2 2 $num_parameters 2000 $limit $interval
		# With strong mutation
		#./MO_GOMEA -r -M 2 2 $num_parameters 2000 $limit $interval
		# With stopping small populations
		#./MO_GOMEA -r -z 2 2 $num_parameters 2000 $limit $interval
		for f in elitist*.dat;
			do mv $f ${num_parameters}_${i}_${f};
		done
	done
done

