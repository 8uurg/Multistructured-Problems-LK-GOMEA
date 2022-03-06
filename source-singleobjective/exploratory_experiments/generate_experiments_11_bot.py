from itertools import product
from functools import reduce

def prod(x):
    return reduce(lambda a, b: a * b, x, 1)

# This experiment investigates the ability
exp_name = "eoa2_bot"

# One Hour!
time_limit = 60 * 60 # s

# This python script generates an `experiments.txt`
# a cross product for the following specification:

# Function Parameters
# - n, k and fn
ns = (25, 50, 100, 200, 400)
ks = (5,)
fns = (1, 2, 4, 8, 16)

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
population_sizes = ("IMR",)

# - Topologies & FOS & multifos
# Format: (topology, distance-metric, fos-type, multifos)
t_f_mfs = (
    ("0", "2", "3", "0"), # Standard GOMEA
    ("7_-1", "2", "3", "1"), # Kernel-GOMEA
    ("15_-1", "2", "3", "2"), # Cluster/Kernel-GOMEA
    ("17_-1", "2", "3", "2") # Cluster-GOMEA
)

approaches = product(seeds, population_sizes, t_f_mfs)
len_approaches = prod(len(k) for k in (population_sizes, t_f_mfs))

print(f"{len_approaches} approaches x {len(seeds)} repetitions = {len_approaches * len(seeds)} runs / instance")

print(f"Total: {len_functions * len_approaches * len(seeds)} runs.")


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

    population_sizing_scheme = 0
    population_size = 4
    
    if isinstance(population_type, int):
        # Fixed Population Size
        population_sizing_scheme = 3

        population_size = population_type
    elif population_type == "IMS":
        population_sizing_scheme = 0
    elif population_type == "SPS":
        population_sizing_scheme = -1
    elif population_type == "IMR":
        population_sizing_scheme = -2
    else:
        raise RuntimeError("Unknown population type specified.")
    
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
        f"--scheme={population_sizing_scheme} "
        f"--populationSize={population_size} "
        f"--FOS={fostype} "
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