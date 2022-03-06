# NA (ran with use_single_objective_directions = false, skips domination style acceptance step entirely, single objective or not)
./build/MO_GOMEA -a3 -oresults/m_100_0_vs_o/line_clustering_scalarstep_kernel_nodomination_512_3 -s512_3 -imaxcut/maxcut_instance_100_0.txt 6 2 100 1000 1000000 10000

# use_single_objective_directions = false (note, only approach 1 & 2 should be affected)
./build/MO_GOMEA -a4 -oresults/m_100_0_vs_o/line_clustering_scalarstep_cluster_512_3 -s512_3 -imaxcut/maxcut_instance_100_0.txt 6 2 100 1000 1000000 10000
./build/MO_GOMEA -a5 -oresults/m_100_0_vs_o/line_clustering_scalarstep_kernel_512_3 -s512_3 -imaxcut/maxcut_instance_100_0.txt 6 2 100 1000 1000000 10000

# use_single_objective_directions = true
# Approach 0 always uses single objective directions -- but was ran with use_single_objective_directions = false.
./build/MO_GOMEA -a0 -oresults/m_100_0_vs_o/euclidean_clustering_512_3 -s512_3 -imaxcut/maxcut_instance_100_0.txt 6 2 100 1000 1000000 10000
./build/MO_GOMEA -a1 -oresults/m_100_0_vs_o/line_clustering_scalarstep_cluster_domination_or_single_512_3 -s512_3 -imaxcut/maxcut_instance_100_0.txt 6 2 100 1000 1000000 10000
./build/MO_GOMEA -a2 -oresults/m_100_0_vs_o/line_clustering_scalarstep_kernel_domination_or_single_512_3 -s512_3 -imaxcut/maxcut_instance_100_0.txt 6 2 100 1000 1000000 10000

# 5 clusters
# NA (ran with use_single_objective_directions = false, skips domination style acceptance step entirely, single objective or not)
./build/MO_GOMEA -a3 -oresults/m_100_0_vs_o/line_clustering_scalarstep_kernel_nodomination_512_5 -s512_5 -imaxcut/maxcut_instance_100_0.txt 6 2 100 1000 1000000 10000

# use_single_objective_directions = false (note, only approach 1 & 2 should be affected)
./build/MO_GOMEA -a4 -oresults/m_100_0_vs_o/line_clustering_scalarstep_cluster_512_5 -s512_5 -imaxcut/maxcut_instance_100_0.txt 6 2 100 1000 1000000 10000
./build/MO_GOMEA -a5 -oresults/m_100_0_vs_o/line_clustering_scalarstep_kernel_512_5 -s512_5 -imaxcut/maxcut_instance_100_0.txt 6 2 100 1000 1000000 10000

# use_single_objective_directions = true
# Approach 0 always uses single objective directions -- but was ran with use_single_objective_directions = false.
./build/MO_GOMEA -a0 -oresults/m_100_0_vs_o/euclidean_clustering_512_5 -s512_5 -imaxcut/maxcut_instance_100_0.txt 6 2 100 1000 1000000 10000
./build/MO_GOMEA -a1 -oresults/m_100_0_vs_o/line_clustering_scalarstep_cluster_domination_or_single_512_5 -s512_5 -imaxcut/maxcut_instance_100_0.txt 6 2 100 1000 1000000 10000
./build/MO_GOMEA -a2 -oresults/m_100_0_vs_o/line_clustering_scalarstep_kernel_domination_or_single_512_5 -s512_5 -imaxcut/maxcut_instance_100_0.txt 6 2 100 1000 1000000 10000