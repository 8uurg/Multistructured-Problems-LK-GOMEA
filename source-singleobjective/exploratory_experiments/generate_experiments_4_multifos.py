from itertools import product

# This experiment simply experiments with tuning the topology parameter.
exp_name = "multiplemodels"

time_limit = 1 * 60 # s

# This python script generates an `experiments.txt`
# a cross product for the following specification:

# - n, k and s
n_k_and_s = (
    (25, 5, 5),
)

# - 100 seeds (42 to 142 (inclusive))
seeds = range(42, 142+1)

# - Population Sizes
population_sizes = (16, 32, 64, 128, 256, 512)

# - Topologies
topologies = (
    "0", 
    "1_1_0", 
    "6_5_0", "6_10_0", "6_20_0", "6_40_0", "6_80_0",
    "7_5_0", "7_10_0", "7_20_0", "7_40_0", "7_80_0")

# - Multiple FOSs per population?
#   > No, just 1 <0>
#   > Yes, 1 per individual <1>
multifoss = ("0", "1")

# Generate a single command from a specification iterable.
def generate_experiment(spec):
    (n, k, s), population_size, seed, topology, multifos = spec

    vtr = n
    vtr_n_unique = 2
    
    folder = (
        f"results/"
        f"{exp_name}/"
        f"n{n}__k{k}__s{s}__pop{population_size}__seed{seed}__"
        f"topology{topology}__"
        f"multifos{multifos}"
    )
    
    return (
        "./build/GOMEA "
        "--approach=1 "
        f"--L={n} "
        f"--problem=13_{k}_{s} "
        f"--timeLimit={time_limit} "
        "--maxEvals=10000000 "
        "--alphabet=2 "
        f"--folder={folder} "
        f"--crowding=0 "
        f"--fitness_sharing=0 "
        f"--seed={seed} "
        f"--topology={topology} "
        "--nfcombine=-2 "
        "--donorSearch "
        # "--scheme=0 "
        "--scheme=3 "
        f"--populationSize={population_size} "
        "--mixingOperator=0 "
        f"--vtr={vtr}_{vtr_n_unique} "
        "--noveltyArchive=0.0_0 "
        f"--multifos={multifos} "
        "--v=0"
    )

# Generate file.
with open('experiments.txt', 'w') as f:
    # Spec nested is iter[((n, k), seed, topology)]
    for spec in product(n_k_and_s, population_sizes, seeds, topologies, multifoss):
        f.write(generate_experiment(spec))
        f.write('\n')