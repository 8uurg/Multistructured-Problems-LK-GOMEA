from itertools import product

# This python script generates an `experiments.txt`
# a cross product for the following specification:

# - Size and k 
#   > n =  100; k = 1
#   > n =  100; k = 2
#   > n =  100; k = 4
#   > n =  100; k = 8
#   > n =  500; k = 1
#   > n =  500; k = 2
#   > n =  500; k = 4
#   > n =  500; k = 8
#   > n =  500; k = 10
#   > n = 1000; k = 1
#   > n = 1000; k = 2
#   > n = 1000; k = 4
#   > n = 1000; k = 8
#   > n = 1000; k = 10
#   > n = 1000; k = 11
n_and_ks = (
    ( 100, 1), ( 100, 2), ( 100, 4), ( 100, 8),
    ( 500, 1), ( 500, 2), ( 500, 4), ( 500, 8), ( 500, 10),
    (1000, 1), (1000, 2), (1000, 4), (1000, 8), (1000, 10), (1000, 11)
)

# - 20 seeds (42 to 62 (inclusive))
seeds = range(42, 62+1)

# - Crowding:
#   > No Crowding <0> 
#   > Crowding (Replace Nearest) <2>
crowdings = ("0", "2")

# - Fitness Sharing:
#   > No Fitness Sharing <0>
#   > Fitness Sharing (Include Parent)
#      > Sigma =  5; Alpha = 2; Fitness Filter = Y; Steady State = N; <2_5_2_1_0>
#      > Sigma =  5; Alpha = 2; Fitness Filter = N; Steady State = N; <2_5_2_0_0>
#      > Sigma =  5; Alpha = 2; Fitness Filter = Y; Steady State = Y; <2_5_2_1_1>
#      > Sigma =  5; Alpha = 2; Fitness Filter = Y; Steady State = Y; <2_5_2_0_1>
#      > Sigma = 10; Alpha = 2; Fitness Filter = Y; Steady State = N; <2_10_2_1_0>
#      > Sigma = 15; Alpha = 2; Fitness Filter = Y; Steady State = N; <2_15_2_1_0>
#      > Sigma = 15; Alpha = 4; Fitness Filter = Y; Steady State = N; <2_15_4_1_0>
#      > Sigma = 15; Alpha = 8; Fitness Filter = Y; Steady State = N; <2_15_8_1_0>
#   > Fitness Sharing (Exclude Parent)
#      > Sigma =  5; Alpha = 2; Fitness Filter = Y; Steady State = N; <1_5_2_1_0>
fitness_sharings = (
    "0", 
    "2_5_2_1_0", "2_5_2_0_0", "2_5_2_0_1", "2_5_2_1_1", "2_10_2_1_0", "2_15_2_1_0", "2_15_4_1_0", "2_15_8_1_0",
    "1_5_2_1_0"
)

# Generate a single command from a specification iterable.
def generate_experiment(spec):
    (n, k), seed, crowding, fitness_sharing = spec

    vtr = n
    vtr_n_unique = 2*k
    
    folder = (
        f"results/standard/"
        f"n{n}__k{k}__seed{seed}__"
        f"crowding{crowding}__fitness_sharing{fitness_sharing}"
    )
    
    return (
        "./build/GOMEA "
        "--approach=1 "
        f"--L={n} "
        f"--problem=12_{k}_0 "
        "--timeLimit=3600 "
        "--maxEvals=10000000 "
        "--alphabet=2 "
        f"--folder={folder} "
        f"--crowding={crowding} "
        f"--fitness_sharing={fitness_sharing} "
        f"--seed={seed} "
        "--nfcombine=-2 "
        "--donorSearch "
        "--scheme=0 "
        "--mixingOperator=0 "
        f"--vtr={vtr}_{vtr_n_unique} "
        "--noveltyArchive=0.0_0 "
        "--v=0"
    )

# Generate file.
with open('experiments.txt', 'w') as f:
    # Spec nested is iter[((n, k), seed, crowding, fitness_sharing)]
    for spec in product(n_and_ks, seeds, crowdings, fitness_sharings):
        f.write(generate_experiment(spec))
        f.write('\n')