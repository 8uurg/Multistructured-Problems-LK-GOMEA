from itertools import product

# This experiment simply experiments with tuning the topology parameter.
exp_name = "topologiesmore"

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

# - 1000 seeds (42 to 1042 (inclusive))
seeds = range(42, 1042+1)

# - a few topologies
#   > Default (full) <0>
#   > Nearest Better Clustering, Q3 + 1.5 * IQR boundary, exclude zeros
#   > Nearest Better Clustering, Q3 + 1.5 * IQR boundary, include zeros
topologies = ("0", "1_1_0", "1_1_1")

# Generate a single command from a specification iterable.
def generate_experiment(spec):
    (n, k), seed, topology = spec

    vtr = n
    vtr_n_unique = 2*k
    
    folder = (
        f"results/"
        f"{exp_name}/"
        f"n{n}__k{k}__seed{seed}__"
        f"topology{topology}"
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
        f"--crowding=0 "
        f"--fitness_sharing=0 "
        f"--seed={seed} "
        f"--topology={topology} "
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
    # Spec nested is iter[((n, k), seed, topology)]
    for spec in product(n_and_ks, seeds, topologies):
        f.write(generate_experiment(spec))
        f.write('\n')