from itertools import product, chain, repeat
import math

exp_name = "paper"

gomea_path = "./build/GOMEA"
gomea_approaches = [
    ("GOMEA", "1 --donorSearch"), 
    ("Kernel GOMEA - Asymmetric", "1 --topology=6_-1 --multifos=1 --donorSearch"),
    ("Kernel GOMEA - Symmetric","1 --topology=7_-1 --multifos=1 --donorSearch"),
]
dsmga2_directory = "DSMGA-II-TwoEdge-master-PopFree"
dsmga2_path = "./build/DSMGA2IMS"

# Repeat a few times for good measure.
number_of_runs = 30
run_start_index = 0
# Use population sizing scheme.
population_size = -1
# Use the default setting for the number of clusters:
# Start at 3, for each doubling of the population size, increase by one.
num_clust = 0
# 
ls_bot = [10, 20, 40, 80, 160, 320]
ls_maxcut = [6, 12, 25, 50, 100]
#
max_number_of_evaluations = 100_000_000
# Should have converged within this time limit. Otherwise thinigs
# will take too long...
timeout_time = "6h"
time_limit = 6 * 60 * 60 # seconds
search_space_multiplier = 8
capped_l = math.log2(max_number_of_evaluations / search_space_multiplier)

def get_number_of_evaluations(l: int):
    return max_number_of_evaluations
    # if l >= capped_l: return max_number_of_evaluations
    # return 2**(l) * 4

def get_seed(run_idx: int):
    return 42 + run_idx

# Flags for enabling / disabling particular subexperiments.
run_onemax = False
run_maxcut = True
run_bot    = True

file_mode = 'w'

# 1. Onemax vs Zeromax
if run_onemax:
    # Not implemented: we are not performing this experiment for the paper
    # right now. Implement when we have time left over.
    pass
    # file_mode = 'a'

# 2. MaxCut

# Format: (n, maxcut_instance)
def get_ns_and_instances_maxcut():
    return (
        (l, f"./data/maxcut/maxcut_instance_{l}_0.txt")
        for l in ls_maxcut
    )

if run_maxcut:
    with open('experiments.txt', file_mode) as f:
        # GOMEA
        for (human_a, a), run_idx, (l, maxcut_instance) in product(gomea_approaches, range(run_start_index, run_start_index + number_of_runs), get_ns_and_instances_maxcut()):
            # Avoid giving small instances too many evaluations.
            number_of_evaluations = get_number_of_evaluations(l)
            vtr = -1.0
            with open(maxcut_instance + ".vtr") as vtrf:
                vtr = float(vtrf.readline())
            f.write((
                f"timeout {timeout_time} "
                f"{gomea_path} "
                f"--approach={a} " # approach
                f"--folder=results/{exp_name}_maxcut/approach_{human_a.replace(' ', '-')}__run_{run_idx}__l_{l} " # output folder
                f"--scheme=0 " # population size & number of clusters
                "--problem=3 " # problem index
                "--alphabet=2 "
                f"--L={l} " # number of parameters
                f"--instance={maxcut_instance} " # problem instance
                f"--vtr={vtr} " # value to reach
                f"--seed={get_seed(run_idx)}"
                f"--maxEvals={number_of_evaluations} " # evaluation budget
                f"--timeLimit={time_limit} "
                " || true"
                "\n"))
        # DSMGA-II with IMS
        for run_idx, (l, maxcut_instance) in product(range(run_start_index, run_start_index + number_of_runs), get_ns_and_instances_maxcut()):
            #
            number_of_evaluations = get_number_of_evaluations(l)
            vtr = -1.0
            with open(maxcut_instance + ".vtr") as vtrf:
                vtr = float(vtrf.readline())
            #
            initial_population_size = 4
            maximum_number_of_generations = 200 # Same as GOMEA
            f.write((
                f"cd {dsmga2_directory} && "
                f"timeout {timeout_time} "
                f"{dsmga2_path} "
                f"{l} {initial_population_size} "
                "3 " # problem
                f"{maximum_number_of_generations} {number_of_evaluations} {time_limit} "
                f"{vtr} "
                f"{get_seed(run_idx)} "
                f"../results/{exp_name}_maxcut/approach_DSMGA2__run_{run_idx}__l_{l} " # folder
                f"../{maxcut_instance} " # instance
                " || true"
                "\n"
            ))
            
    file_mode = 'a'

# 3. Best-of-Traps
fnss = [1, 2, 4, 8] # Remove 16 from [1, 2, 4, 8, 16]
# As 8 is usually already difficult enough.

# Format: (n, fns, bot_instance_0)
def get_ns_and_instances_bot():
    return chain.from_iterable(
        ((n, fns, f"./data/bot/bot_n{n}k5fns{fns}s42.txt") for fns in fnss) 
        for n in ls_bot
    )
if run_bot:
    with open('experiments.txt', file_mode) as f:
        for (human_a, a), run_idx, (l, fns, bot_instance) in product(gomea_approaches, range(run_start_index, run_start_index + number_of_runs), get_ns_and_instances_bot()):
            # Avoid giving small instances too many evaluations.
            number_of_evaluations = get_number_of_evaluations(l)
            vtr = f"{l}"
            f.write((
                f"timeout {timeout_time} "
                f"{gomea_path} "
                f"--approach={a} " # approach
                f"--folder=results/{exp_name}_bot/approach_{human_a.replace(' ', '-')}__run_{run_idx}__l_{l}__fns_{fns} " # output folder
                f"--scheme=0 " # population size & number of clusters
                "--problem=15 " # problem index
                "--alphabet=2 "
                f"--L={l} " # number of parameters
                f"--instance=f_{bot_instance} " # problem instance
                f"--vtr={vtr} " # value to reach
                f"--seed={get_seed(run_idx)}"
                f"--maxEvals={number_of_evaluations} " # evaluation budget
                f"--timeLimit={time_limit} "
                " || true"
                "\n"))
        # DSMGA-II with IMS
        for run_idx, (l, fns, bot_instance) in product(range(run_start_index, run_start_index + number_of_runs), get_ns_and_instances_bot()):
            #
            number_of_evaluations = get_number_of_evaluations(l)
            vtr = f"{l}"
            initial_population_size = 4
            maximum_number_of_generations = 200 # Same as GOMEA
            f.write((
                f"cd {dsmga2_directory} && "
                f"timeout {timeout_time} "
                f"{dsmga2_path} "
                f"{l} {initial_population_size} "
                "10 " # problem
                f"{maximum_number_of_generations} {number_of_evaluations} {time_limit} "
                f"{vtr} "
                f"{get_seed(run_idx)} "
                f"../results/{exp_name}_bot/approach_DSMGA2__run_{run_idx}__l_{l}__fns_{fns} " # folder
                f"f_../{bot_instance} " # instance
                " || true"
                "\n"
            ))
    file_mode = 'a'
