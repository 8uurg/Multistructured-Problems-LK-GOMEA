from itertools import product
from functools import reduce
from pathlib import Path
import re
import sys
import math

def prod(x):
    return reduce(lambda a, b: a * b, x, 1)

# This experiment is for random nkq landscapes (ran at the end of April)
exp_name = "maxsat_nb"

# One minute
time_limit = 1 * 60 # s

# This python script generates an `experiments.txt`
# a cross product for the following specification:

# n = k?
# List of (instance path, string length)
instances = [
    ("./data/sat/instances/dsmga/SAT/uf200/uf200-01.cnf", 200)
]
functions = instances
len_functions = len(instances)

print(f"{len_functions} problems")

# # # Approach configuration.

# # - 10 seeds (42 to 52 (inclusive))
seeds = range(42, 52+1)

# # - Population Sizes, number = fixed population size
population_sizes = (16, 32, 64, 128, 256, 512, 1024, 2048)
# population_sizes = ("IMR",)

# # - Topologies & FOS & multifos
# # Format: (topology, distance-metric, fos-type, multifos)
t_f_mfs = (
    ("0", "2", "3", "0"), # Standard GOMEA
    ("7_-1", "2", "3", "1"), # Kernel-GOMEA
    ("15_-1", "2", "3", "2"), # Cluster/Kernel-GOMEA
    ("17_-1", "2", "3", "2") # Cluster-GOMEA
)
hillclimbers = (0, 1)
sample_refs = (0.05, 0.10, 0.15, 0.20)

approaches = product(seeds, population_sizes, t_f_mfs, hillclimbers, sample_refs)
len_approaches = prod(len(k) for k in (population_sizes, t_f_mfs))
print(f"{len_approaches} approaches x {len(seeds)} repetitions = {len_approaches * len(seeds)} runs / instance")

print(f"Total: {len_functions * len_approaches * len(seeds)} runs.")

# Generate a single command from a specification iterable.
def generate_experiment(spec):
    (problem, approach) = spec
    # print(f"Problem: {problem}")
    # print(f"Approach: {approach}")
    (p, L) = problem
    (seed, population_type, (topology, distance, fostype, multifos), hc, sampleref) = approach

    # Problem under test is 14_{k}_{fn_seed}_{fn}
    # vtr is already set
    vtr = 0.0
    vtr_n_unique = 1

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
        f"L{L}__"
        f"pop{population_type}__seed{seed}__"
        f"d{distance}__"
        f"topology{topology}__"
        f"fos{fostype}__"
        f"multifos{multifos}__"
        f"hc{hc}__"
        f"sampleref{sampleref}"
    )
    
    return (
        "timeout "
        f"-k {time_limit * 2} "
        f"{time_limit} "
        "./build/GOMEA "
        "--approach=1 "
        f"--L={L} "
        f"--problem=8 "
        f"--instance={p} "
        f"--timeLimit={time_limit} "
        "--maxEvals=1000000 "
        "--alphabet=2 "
        f"--folder={folder} "
        f"--seed={seed} "
        f"--topology={topology} "
        # "--donorSearch "
        "--hillClimber={hc} "
        f"--scheme={population_sizing_scheme} "
        f"--populationSize={population_size} "
        f"--FOS={fostype} "
        f"--distance={distance} "
        f"--vtr={vtr}_{vtr_n_unique} "
        f"--multifos={multifos} "
        "--reference=00110010111000111110000000000111101001001111111011000000110111111100110000101010011100100000100010011011001101000100101000100010001101110001000001001100111110011010000001110010001011100011011110100010 "
        f"--sampleref={sampleref} "
        "--v=0 "
        f" || ( echo {folder} failed && exit 9 )"
    )

# Generate file.
with open('experiments.txt', 'w') as f:
    # spec is (function, approach), which splits up in
    # function: (n, k, fn, fn_seed)
    # approach: (seed, population_type, topology, distance, fostype, multifos)
    # num = 0
    for spec in product(functions, approaches):
        f.write(generate_experiment(spec))
        f.write('\n')
        # num += 1
        # if num > 10:
        #     break