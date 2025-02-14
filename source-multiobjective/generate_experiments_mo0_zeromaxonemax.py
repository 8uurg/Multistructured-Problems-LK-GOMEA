#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
#
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
#
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

from itertools import product
from functools import reduce

def prod(x):
    return reduce(lambda a, b: a * b, x, 1)

# This experiment is for MO Scalability on ZeroMax-OneMax
exp_name = "mos_zeromaxonemax"

# 5 minutes
time_limit = 5 * 60 # s
eval_limit = 1000000

archive_size = 1000
log_interval_eval = eval_limit

# This python script generates an `experiments.txt`
# a cross product for the following specification:

problem_idx = 0
functions = [
    (5,),
    (10,),
    (20,),
    (25,),
    (50,),
    (75,),
    (100,),
    (200,),
]
len_functions = len(functions)
num_obj = 2

print(f"{len_functions} problems")

# # # Approach configuration.

# # - 10 seeds (42 to 52 (inclusive)) -- Note: MO GOMEA uses the time as seed...
seeds = range(42, 142)

# # - Population Sizes, number = fixed population size
# population_sizes = (16, 32, 64, 128, 256, 512, 1024, 2048)
# population_sizes = ("IMR",)

# # - Topologies & FOS & multifos
# # Format: (topology, distance-metric, fos-type, multifos)
kernel_mode_and_name = (
    ("", "clustering"),
    ("-k-1_0_0 ", "kernel-sqrtn-assym-euclidean"),
    ("-k-2_0_0 ", "kernel-log2n-assym-euclidean"),
    ("-k-1_1_0 ", "kernel-sqrtn-sym-euclidean"),
    ("-k-2_1_0 ", "kernel-log2n-sym-euclidean"),
    ("-k-1_0_1 ", "kernel-sqrtn-assym-cosine"),
    ("-k-2_0_1 ", "kernel-log2n-assym-cosine"),
    ("-k-1_1_1 ", "kernel-sqrtn-sym-cosine"),
    ("-k-2_1_1 ", "kernel-log2n-sym-cosine"),
)

approaches = product(seeds, kernel_mode_and_name)
len_approaches = prod(len(k) for k in (kernel_mode_and_name,))
print(f"{len_approaches} approaches x {len(seeds)} repetitions = {len_approaches * len(seeds)} runs / instance")

print(f"Total: {len_functions * len_approaches * len(seeds)} runs.")

# Generate a single command from a specification iterable.
def generate_experiment(spec):
    (problem, approach) = spec
    # print(f"Problem: {problem}")
    # print(f"Approach: {approach}")
    (L,) = problem
    (seed, (kernel_mode, kernel_mode_name)) = approach
    
    folder = (
        f"results/"
        f"{exp_name}/"
        f"L{L}__"
        f"seed{seed}__"
        f"cm_{kernel_mode_name}"
    )
    
    return (
        "timeout "
        f"-k {time_limit * 2} "
        f"{time_limit} "
        "./build/MO_GOMEA "
        f"-o{folder} "
        f"{kernel_mode}" 
        f"{problem_idx} "
        f"{num_obj} "
        f"{L} "
        f"{archive_size} "
        f"{eval_limit} "
        f"{log_interval_eval} " 
        f" || ( echo {folder} failed && exit 9 )"
    )

# Generate file.
with open('experiments.txt', 'a') as f:
    for spec in product(functions, approaches):
        f.write(generate_experiment(spec))
        f.write('\n')
        # num += 1
        # if num > 10:
        #     break