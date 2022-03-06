from itertools import product
from functools import reduce

def prod(x):
    return reduce(lambda a, b: a * b, x, 1)

# This experiment investigates the ability
exp_name = "sppopsampling"

# Two minutes should be sufficient, if it takes longer...
time_limit = 2 * 60 # s

# This python script generates an `experiments.txt`
# a cross product for the following specification:

# Function Parameters
# - n, k and fn
ns = (25, 50, )
ks = (5,)
fns = (1, 2, 4)

# - 1 function seeds
ss = (42,)#range(42, 42+5+1)

# Tuples of (n, k, fn, s)
functions = product(ns, ks, fns, ss)
len_functions = prod(len(k) for k in (ns, ks, fns, ss))

print(f"{len_functions} problems")

# # Approach configuration.

# - 10 seeds (42 to 52 (inclusive))
seeds = range(42, 52+1)

# - Population Sizes, number = fixed population size
population_sizes = (16, 32, 64, 128, 256, 512, 1024, 2048)

# - Topologies & FOS & multifos
# Format: (topology, distance-metric, fos-type, multifos)
t_f_mfs = (
    (       "7_40", "3",  "3", "3"), # kNN mating restriction (\w k=40), linkage tree, FOS per node & sampling
    (       "7_-1", "3",  "3", "3"), # kNN mating restriction (\w k=sqrt(|P|), linkage tree, FOS per node & sampling
    (       "7_-2", "3",  "3", "3"), # kNN mating restriction (\w k=log2(|P|), linkage tree, FOS per node & sampling
    ("13_0_0_0.40", "3",  "3", "3"), # Leader Clustering (\w threshold 0.4) mating restriction, Linkage Tree, FOS per cluster & sampling
    # TODO: Leader Clustering with adaptive threshold?
    (      "14_-1", "3",  "3", "3"), # Nearest-Farthest Clustering (\w k=sqrt(|P|)) mating restriction, Linkage Tree, FOS per cluster & sampling
    (      "14_-2", "3",  "3", "3"), # Nearest-Farthest Clustering (\w k=log2(|P|)) mating restriction, Linkage Tree, FOS per cluster & sampling
)

approaches = product(seeds, population_sizes, t_f_mfs)
len_approaches = prod(len(k) for k in (seeds, population_sizes, t_f_mfs))
print(f"{len_approaches} approaches")

# Generate a single command from a specification iterable.
def generate_experiment(spec):
    (problem, approach) = spec
    # print(f"Problem: {problem}")
    # print(f"Approach: {approach}")
    (n, k, fn, fn_seed) = problem
    (seed, population_type, (topology, distance, fostype, multifos)) = approach

    # Problem under test is 14_{k}_{fn_seed}_{fn}
    vtr = n
    # Note: negative is used to indicate to use the built-in points of interest
    # detection. This overrides the neccesity of the variable above.
    vtr_n_unique = -fn

    fos_param = 3
    population_size = 4
    if fostype == "IMS":
        fos_param = 0
    if isinstance(population_type, int):
        population_size = population_type
    
    folder = (
        f"results/"
        f"{exp_name}/"
        f"n{n}__k{k}__s{fn_seed}__fn{fn}__"
        f"pop{population_type}__seed{seed}__"
        f"d{distance}__"
        f"topology{topology}__"
        f"fos{fostype}__"
        f"multifos{multifos}"
    )
    
    return (
        "timeout "
        f"-k {time_limit * 2} "
        f"{time_limit} "
        "./build/GOMEA "
        "--approach=1 "
        f"--L={n} "
        f"--problem=14_{k}_{fn_seed}_{fn} "
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
        "--hillClimber=1 "
        # "--scheme=0 "
        f"--scheme={fos_param} "
        f"--FOS={fostype} "
        f"--populationSize={population_size} "
        f"--distance={distance} "
        "--mixingOperator=0 "
        f"--vtr={vtr}_{vtr_n_unique} "
        "--noveltyArchive=0.0_0 "
        f"--multifos={multifos} "
        "--v=0"
    )

# Generate file.
with open('experiments.txt', 'w') as f:
    # spec is (function, approach), which splits up in
    # function: (n, k, fn, fn_seed)
    # approach: (seed, population_type, topology, distance, fostype, multifos)
    for spec in product(functions, approaches):
        f.write(generate_experiment(spec))
        f.write('\n')