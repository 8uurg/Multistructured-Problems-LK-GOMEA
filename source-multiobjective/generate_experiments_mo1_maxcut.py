from itertools import product
from functools import reduce

def prod(x):
    return reduce(lambda a, b: a * b, x, 1)

# This experiment is for MO - Maxcut
exp_name = "mo_maxcut"

# 1 minute
time_limit = 1 * 60 # s
eval_limit = 1000000

archive_size = 1000
log_interval_eval = eval_limit // 1000

# This python script generates an `experiments.txt`
# a cross product for the following specification:

problem_idx = 4
functions = [
    (6,),
    (12,),
    (25,),
    (50,),
    (100,)
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
    ("-k1_-1_0_0 ", "kernel-sqrtn-assym-euclidean"),
    ("-k1_-2_0_0 ", "kernel-log2n-assym-euclidean"),
    ("-k1_-1_1_0 ", "kernel-sqrtn-sym-euclidean"),
    ("-k1_-2_1_0 ", "kernel-log2n-sym-euclidean"),
    ("-k1_-1_0_1 ", "kernel-sqrtn-assym-cosine"),
    ("-k1_-2_0_1 ", "kernel-log2n-assym-cosine"),
    ("-k1_-1_1_1 ", "kernel-sqrtn-sym-cosine"),
    ("-k1_-2_1_1 ", "kernel-log2n-sym-cosine"),
    ("-k2_-1_0_0 ", "hybrid-sqrtn-assym-euclidean"),
    ("-k2_-2_0_0 ", "hybrid-log2n-assym-euclidean"),
    ("-k2_-1_1_0 ", "hybrid-sqrtn-sym-euclidean"),
    ("-k2_-2_1_0 ", "hybrid-log2n-sym-euclidean"),
    ("-k2_-1_0_1 ", "hybrid-sqrtn-assym-cosine"),
    ("-k2_-2_0_1 ", "hybrid-log2n-assym-cosine"),
    ("-k2_-1_1_1 ", "hybrid-sqrtn-sym-cosine"),
    ("-k2_-2_1_1 ", "hybrid-log2n-sym-cosine"),
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
with open('experiments.txt', 'w') as f:
    for spec in product(functions, approaches):
        f.write(generate_experiment(spec))
        f.write('\n')
        # num += 1
        # if num > 10:
        #     break