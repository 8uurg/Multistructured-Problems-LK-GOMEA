from itertools import product
from functools import reduce
from pathlib import Path
import re
import sys
import math

def prod(x):
    return reduce(lambda a, b: a * b, x, 1)

# This experiment is for random nkq landscapes (ran at the end of April)
exp_name = "eoa2_maxcut"

# One Hour!
time_limit = 60 * 60 # s

# This python script generates an `experiments.txt`
# a cross product for the following specification:

# Find out about the instances!

L_and_idx_extractor = re.compile(r"^n([0-9]+)i([0-9]+).txt$")

def extract_info(p: Path):
    # Last part contains the index: 10 files numbered 1-10; 
    # Second to last part declares the string length "L{length}";
    # Third to last part declares, n and s; 
    L_and_idx_match = L_and_idx_extractor.match(p.parts[-1])
    L = int(L_and_idx_match.group(1))
    idx = int(L_and_idx_match.group(2))

    # Get vtr file and parse it (and subtract epsilon)
    p_vtr = p.parent / f"{p.stem}.bkv"
    vtr = 0.0
    with open(p_vtr, 'r') as f:
        vtr = float(f.readline())
        # Correct for rounding error.
        vtr -= sys.float_info.epsilon * 1e5 * math.log2(vtr)

    return (p, idx, L, vtr)

def instance_filter(instance):
    (p, idx, L, vtr) = instance
    accepted = idx <= 5
    # print(f"{instance}: {accepted}")
    return accepted

# n = k?
# List of (instance path, index, string length, n, s)
instances = list(filter(instance_filter, map(extract_info, Path("./data/maxcut/set0a").glob("*.txt"))))
functions = instances
len_functions = len(instances)

print(f"{len_functions} problems")

# # # Approach configuration.

# # - 10 seeds (42 to 52 (inclusive))
seeds = range(42, 52+1)

# # - Population Sizes, number = fixed population size
# population_sizes = (16, 32, 64, 128, 256, 512, 1024, 2048)
population_sizes = ("IMR",)

# # - Topologies & FOS & multifos
# # Format: (topology, distance-metric, fos-type, multifos)
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
    (p, idx, L, vtr) = problem
    (seed, population_type, (topology, distance, fostype, multifos)) = approach

    # Problem under test is 14_{k}_{fn_seed}_{fn}
    # vtr is already set
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
        f"L{L}__idx{idx}__"
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
        f"--L={L} "
        f"--problem=3 "
        f"--instance={p} "
        f"--timeLimit={time_limit} "
        "--maxEvals=100000000 "
        "--alphabet=2 "
        f"--folder={folder} "
        f"--seed={seed} "
        f"--topology={topology} "
        "--donorSearch "
        # "--hillClimber=1 "
        f"--scheme={population_sizing_scheme} "
        f"--populationSize={population_size} "
        f"--FOS={fostype} "
        f"--distance={distance} "
        f"--vtr={vtr}_{vtr_n_unique} "
        f"--multifos={multifos} "
        "--v=0 "
        f" || ( echo {folder} failed && exit 9 )"
    )

# Generate file.
with open('experiments.txt', 'a') as f:
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