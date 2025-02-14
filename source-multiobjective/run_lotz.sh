#!/bin/bash
#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
#
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
#
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

for num_parameters in {25,50,100,200,400}
do
	for i in {1..100}
	do
		# Standard version, no mutation, no stopping small population
		./MO_GOMEA 3 2 $num_parameters 1000 10000000 1000000
		# With weak mutation
		#./MO_GOMEA -m 3 2 $num_parameters 1000 10000000 1000000
		# With strong mutation
		#./MO_GOMEA -M 3 2 $num_parameters 1000 10000000 1000000
		# With stopping small populations
		#./MO_GOMEA -z 3 2 $num_parameters 1000 10000000 1000000
	done
	echo $num_parameters | awk '{printf("%d",$1);}' >> lotz_statistics.dat
	cat "number_of_evaluations_when_all_points_found_${num_parameters}.dat" | awk 'BEGIN{total_eval=0.0;num=0;std=0.0;}{total_eval += $1;num++;eval[num]=$1;}END{avg=total_eval/num; for(j=1;j<=num;j++){std+=(eval[j]-avg)*(eval[j]-avg);} std=sqrt(std/num); printf(" %f %f\n", avg, std);}' >> lotz_statistics.dat
done

