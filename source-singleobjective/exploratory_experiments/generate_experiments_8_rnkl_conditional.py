from itertools import product
from functools import reduce
from pathlib import Path
import re
import sys
import math

def prod(x):
    return reduce(lambda a, b: a * b, x, 1)

# This experiment investigates the ability
exp_name = "randomnklandscapesc"

# Five minutes should be sufficient, if it takes longer...
time_limit = 5 * 60 # s

# This python script generates an `experiments.txt`
# a cross product for the following specification:

# Find out about the instances!

# We do not want the vtr files to be involved directly
def is_not_vtr_file(p: Path):
    return not p.parts[-1].endswith("_vtr.txt")

idx_extractor = re.compile(r"^seed_([0-9]+).txt$")
L_extractor = re.compile(r"^L_([0-9]+)$")
k_extractor = re.compile(r"^k_([0-9]+)$")
Q_extractor = re.compile(r"^Q_([0-9]+)$")

def extract_info(p: Path):
    # Last part contains the index: 10 files numbered 1-10; 
    # Second to last part declares the string length "L{length}";
    # Third to last part declares, n and s; 
    idx = idx_extractor.match(p.parts[-1]).group(1)
    Q = Q_extractor.match(p.parts[-2]).group(1)
    k = k_extractor.match(p.parts[-3]).group(1)
    L = L_extractor.match(p.parts[-4]).group(1)

    # Get vtr file and parse it (and subtract epsilon)
    p_vtr = p.parent / f"{p.stem}_vtr.txt"
    vtr = 0.0
    with open(p_vtr, 'r') as f:
        vtr = float(f.readline())
        # Correct for rounding error.
        vtr -= sys.float_info.epsilon * 1e5 * math.log2(vtr)

    return (p, idx, L, k, Q, vtr)

# n = k?
# List of (instance path, index, string length, n, s)
instances = list(map(extract_info, filter(is_not_vtr_file, Path("./data/nk/instances_random").glob("*/*/*/*.txt"))))
functions = instances
len_functions = len(instances)

print(f"{len_functions} problems")

# # # Approach configuration.

# # - 10 seeds (42 to 52 (inclusive))
seeds = range(42, 52+1)

# # - Population Sizes, number = fixed population size
# population_sizes = (16, 32, 64, 128, 256, 512, 1024, 2048)
population_sizes = ("IMS",)

# # - Topologies & FOS & multifos
# # Format: (topology, distance-metric, fos-type, multifos)
t_f_mfs = (
#    (          "0", "3", "10", "0"), # No mating restrictions, problem specific (global) fos, no multiple fos
#    (          "0", "3",  "3", "0"), # No mating restrictions, linkage tree, no multiple fos
    (          "0", "3",  "4", "0"), # No mating restrictions, random tree, no multiple fos
#     (       "7_40", "3", "10", "0"), # kNN mating restriction (\w k=40), problem specific (global) fos, no multiple fos
#     (       "7_40", "3",  "3", "1"), # kNN mating restriction (\w k=40), linkage tree, FOS per node
    # kNN mating restriction (\w k=sqrt(|P|), linkage tree
#    (       "7_-1", "2", "0", "0"), # Hamming, Global FOS
    (       "7_-1", "2",  "3", "1"), # Hamming, FOS per node
#    (       "7_-1", "3",  "3", "1"), # 'Agreement' Distance, FOS per node
#     (       "7_-2", "3",  "3", "1"), # kNN mating restriction (\w k=log2(|P|), linkage tree, FOS per node
#     ("13_0_0_0.40", "3", "10", "0"), # Leader Clustering (\w threshold 0.4) mating restriction, problem specific (global) fos, no multiple fos
#     ("13_0_0_0.40", "3",  "3", "2"), # Leader Clustering (\w threshold 0.4) mating restriction, Linkage Tree, FOS per cluster
#     # TODO: Leader Clustering with adaptive threshold?
#     (      "14_-1", "3",  "3", "2"), # Nearest-Farthest Clustering (\w k=sqrt(|P|)) mating restriction, Linkage Tree, FOS per cluster
#     (      "14_-2", "3",  "3", "2"), # Nearest-Farthest Clustering (\w k=log2(|P|)) mating restriction, Linkage Tree, FOS per cluster

    # Nearest-Farthest Clustering (\w k=sqrt(|P|)) mating restriction, Linkage Tree, FOS per cluster
    (        "14_-1", "2",  "3", "2"), # Hamming Distance
#    (        "14_-1", "3",  "3", "2"), # 'Agreement' Distance

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
    (p, idx, L, k, Q, vtr) = problem
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
    else:
        raise RuntimeError("Unknown population type specified.")
    
    folder = (
        f"results/"
        f"{exp_name}/"
        f"L{L}__k{k}__Q{Q}__idx{idx}__"
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
        f"--problem=2 "
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
        "--GOM=1 "
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