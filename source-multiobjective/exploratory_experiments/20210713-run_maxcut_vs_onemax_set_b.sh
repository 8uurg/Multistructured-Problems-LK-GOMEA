#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
#
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
#
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

# NA (ran with use_single_objective_directions = false, skips domination style acceptance step entirely, single objective or not)
./build/MO_GOMEA -a3 -oresults/m0b_100_0_vs_o/line_clustering_scalarstep_kernel_nodomination_512_3 -s512_3 -imaxcut_sets/set0b/n0000100i00.txt 6 2 100 1000 1000000 10000

# use_single_objective_directions = false (note, only approach 1 & 2 should be affected)
./build/MO_GOMEA -a4 -oresults/m0b_100_0_vs_o/line_clustering_scalarstep_cluster_512_3 -s512_3 -imaxcut_sets/set0b/n0000100i00.txt 6 2 100 1000 1000000 10000
./build/MO_GOMEA -a5 -oresults/m0b_100_0_vs_o/line_clustering_scalarstep_kernel_512_3 -s512_3 -imaxcut_sets/set0b/n0000100i00.txt 6 2 100 1000 1000000 10000

# use_single_objective_directions = true
# Approach 0 always uses single objective directions -- but was ran with use_single_objective_directions = false.
./build/MO_GOMEA -a0 -oresults/m0b_100_0_vs_o/euclidean_clustering_512_3 -s512_3 -imaxcut_sets/set0b/n0000100i00.txt 6 2 100 1000 1000000 10000
./build/MO_GOMEA -a1 -oresults/m0b_100_0_vs_o/line_clustering_scalarstep_cluster_domination_or_single_512_3 -s512_3 -imaxcut_sets/set0b/n0000100i00.txt 6 2 100 1000 1000000 10000
./build/MO_GOMEA -a2 -oresults/m0b_100_0_vs_o/line_clustering_scalarstep_kernel_domination_or_single_512_3 -s512_3 -imaxcut_sets/set0b/n0000100i00.txt 6 2 100 1000 1000000 10000

# 5 clusters (instead of 3)
# NA (ran with use_single_objective_directions = false, skips domination style acceptance step entirely, single objective or not)
./build/MO_GOMEA -a3 -oresults/m0b_100_0_vs_o/line_clustering_scalarstep_kernel_nodomination_512_5 -s512_5 -imaxcut_sets/set0b/n0000100i00.txt 6 2 100 1000 1000000 10000

# use_single_objective_directions = false (note, only approach 1 & 2 should be affected)
./build/MO_GOMEA -a4 -oresults/m0b_100_0_vs_o/line_clustering_scalarstep_cluster_512_5 -s512_5 -imaxcut_sets/set0b/n0000100i00.txt 6 2 100 1000 1000000 10000
./build/MO_GOMEA -a5 -oresults/m0b_100_0_vs_o/line_clustering_scalarstep_kernel_512_5 -s512_5 -imaxcut_sets/set0b/n0000100i00.txt 6 2 100 1000 1000000 10000

# use_single_objective_directions = true
# Approach 0 always uses single objective directions -- but was ran with use_single_objective_directions = false.
./build/MO_GOMEA -a0 -oresults/m0b_100_0_vs_o/euclidean_clustering_512_5 -s512_5 -imaxcut_sets/set0b/n0000100i00.txt 6 2 100 1000 1000000 10000
./build/MO_GOMEA -a1 -oresults/m0b_100_0_vs_o/line_clustering_scalarstep_cluster_domination_or_single_512_5 -s512_5 -imaxcut_sets/set0b/n0000100i00.txt 6 2 100 1000 1000000 10000
./build/MO_GOMEA -a2 -oresults/m0b_100_0_vs_o/line_clustering_scalarstep_kernel_domination_or_single_512_5 -s512_5 -imaxcut_sets/set0b/n0000100i00.txt 6 2 100 1000 1000000 10000
