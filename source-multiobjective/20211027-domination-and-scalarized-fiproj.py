from itertools import product, chain, repeat
import math

exp_name = "paper_ds_fiproj"

approaches = [
    "0 -F1", # Original MO-GOMEA
    "0_-1 -c8 -x-1 -F1", # + KNN
    "0_-1 -c9 -x-1 -F1", # + Symmetric KNN (i.e. given solutions `a` and `b`, a is KNN of b, or b is KNN of a)
    "-3_-1 -F1",
    "-3_-1 -c8 -x-1 -F1",
    "-3_-1 -c9 -x-1 -F1",
]

# Repeat a few times for good measure.
# Low for now: 6h each worst case is a lot of time!
number_of_runs = 4
# Use population sizing scheme.
population_size = -1
# Use the default setting for the number of clusters:
# Start at 3, for each doubling of the population size, increase by one.
num_clust = 0
# 
ls = [12, 25, 50, 100]
#
max_number_of_evaluations = 100_000_000
# Should have converged within this time limit. Otherwise thinigs
# will take too long...
timeout_time = "6h"
search_space_multiplier = 8
capped_l = math.log2(max_number_of_evaluations / search_space_multiplier)

def get_number_of_evaluations(l: int):
    if l >= capped_l: return max_number_of_evaluations
    return 2**(l) * 4

# Flags for enabling / disabling particular subexperiments.
run_onemax_vs_zeromax = True
run_maxcut_vs_onemax  = True
run_maxcut_vs_maxcut  = True
run_bot_vs_onemax     = True
run_bot_vs_maxcut     = True
run_bot_vs_bot        = True

file_mode = 'w'

# 1. Onemax vs Zeromax
if run_onemax_vs_zeromax:
    with open('experiments.txt', file_mode) as f:
        for a, run_idx, l in product(approaches, range(number_of_runs), ls):
            # Avoid giving small instances too many evaluations.
            number_of_evaluations = get_number_of_evaluations(l)
            f.write((
                f"timeout {timeout_time} "
                "./build/MO_GOMEA "
                f"-a{a} " # approach
                f"\"-oresults/{exp_name}_onemax_vs_zeromax/approach_{a.replace(' ', '-')}__run_{run_idx}__l_{l}\" " # output folder
                f"-s{population_size}_{num_clust} " # population size & number of clusters
                "0 " # problem index
                "2 " # number of objectives
                f"{l} " # number of parameters
                "1000 " # elitist archive size target
                f"{number_of_evaluations} " # evaluation budget
                f"{number_of_evaluations}"  # logging interval (in #evaluations)
                " || true"
                "\n"))
    file_mode = 'a'


# 2. MaxCut vs OneMax

# Format: (n, maxcut_instance_0)
ns_and_instances_maxcut_2 = (
    (l, f"./maxcut/maxcut_instance_{l}_0.txt")
    for l in ls
)

if run_maxcut_vs_onemax:
    with open('experiments.txt', file_mode) as f:
        for a, run_idx, (l, maxcut_instance_0) in product(approaches, range(number_of_runs), ns_and_instances_maxcut_2):
            # Avoid giving small instances too many evaluations.
            number_of_evaluations = get_number_of_evaluations(l)
            approximation_front_file = f"./reference_fronts/maxcut_onemax/n_{l}__i_0.txt"
            f.write((
                f"timeout {timeout_time} "
                "./build/MO_GOMEA "
                f"-a{a} " # approach
                f"-oresults/{exp_name}_maxcut_vs_onemax/approach_{a.replace(' ', '-')}__run_{run_idx}__l_{l} " # output folder
                f"-s{population_size}_{num_clust} " # population size & number of clusters
                f"-i{maxcut_instance_0} " # problem instance
                f"-f{approximation_front_file} " # front approximation
                "6 " # problem index
                "2 " # number of objectives
                f"{l} " # number of parameters
                "1000 " # elitist archive size target
                f"{number_of_evaluations} " # evaluation budget
                f"{number_of_evaluations}"  # logging interval (in #evaluations)
                " || true"
                "\n"))
    file_mode = 'a'

# 3. MaxCut vs MaxCut
# Format: n
# Note: instance is selected in the MO_GOMEA code itself, problem index 4 does
# not allow for manual selection of instances (unless by overwriting the right file)

if run_maxcut_vs_maxcut:
    with open('experiments.txt', file_mode) as f:
        for a, run_idx, l in product(approaches, range(number_of_runs), ls):
            # Avoid giving small instances too many evaluations.
            number_of_evaluations = get_number_of_evaluations(l)
            f.write((
                f"timeout {timeout_time} "
                "./build/MO_GOMEA "
                f"-a{a} " # approach
                f"-oresults/{exp_name}_maxcut_vs_maxcut/approach_{a.replace(' ', '-')}__run_{run_idx}__l_{l} " # output folder
                f"-s{population_size}_{num_clust} " # population size & number of clusters
                "4 " # problem index
                "2 " # number of objectives
                f"{l} " # number of parameters
                "1000 " # elitist archive size target
                f"{number_of_evaluations} " # evaluation budget
                f"{number_of_evaluations}"  # logging interval (in #evaluations)
                " || true"
                "\n"))
    file_mode = 'a'


# 4. BOT vs OneMax
fnss = [1, 2, 4, 8, 16]
# Format: (n, fns, bot_instance_0)
ns_and_instances_bot_4 = chain.from_iterable(
    ((n, fns, f"./bestoftraps/bot_n{n}k5fns{fns}s0.txt") for fns in fnss) 
    for n in ls
)
if run_bot_vs_onemax:
    with open('experiments.txt', file_mode) as f:
        for a, run_idx, (l, fns, bot_instance_0) in product(approaches, range(number_of_runs), ns_and_instances_bot_4):
            # Avoid giving small instances too many evaluations.
            number_of_evaluations = get_number_of_evaluations(l)
            approximation_front_file = f"./reference_fronts/bot_onemax/n_{l}__fns_{fns}__s_0__k_5.txt"
            f.write((
                f"timeout {timeout_time} "
                "./build/MO_GOMEA "
                f"-a{a} " # approach
                f"-oresults/{exp_name}_bot_vs_onemax/approach_{a.replace(' ', '-')}__run_{run_idx}__l_{l}__fns_{fns} " # output folder
                f"-s{population_size}_{num_clust} " # population size & number of clusters
                f"-if_{bot_instance_0} " # problem instance
                f"-f{approximation_front_file} " # front approximation
                "8 " # problem index
                "2 " # number of objectives
                f"{l} " # number of parameters
                "1000 " # elitist archive size target
                f"{number_of_evaluations} " # evaluation budget
                f"{number_of_evaluations}"  # logging interval (in #evaluations)
                " || true"
                "\n"))
    file_mode = 'a'
# Note: The fronts still have to reconstructed due to the decompostion.

# 5. BOT vs MaxCut
fnss = [1, 2, 4, 8, 16]
# Format: (n, bot_instance_0, maxcut_instance_0)
ns_and_instances_bot_maxcut_5 = ns_and_instances_bot_4 = chain.from_iterable(
    (((n, fns, f"./bestoftraps/bot_n{n}k5fns{fns}s0.txt", f"./maxcut/maxcut_instance_{n}_0.txt") for fns in fnss)) 
    for n in ls
)
if run_bot_vs_maxcut:
    with open('experiments.txt', file_mode) as f:
        for a, run_idx, (l, fns, bot_instance_0, maxcut_instance_0) in product(approaches, range(number_of_runs), ns_and_instances_bot_maxcut_5):
            # Avoid giving small instances too many evaluations.
            number_of_evaluations = get_number_of_evaluations(l)
            approximation_front_file = f"./reference_fronts/bot_maxcut/l_{l}__fns_{fns}__s_0__k_5.txt"
            f.write((
                f"timeout {timeout_time} "
                "./build/MO_GOMEA "
                f"-a{a} " # approach
                f"-oresults/{exp_name}_bot_vs_maxcut/approach_{a.replace(' ', '-')}__run_{run_idx}__l_{l}__fns_{fns} " # output folder
                f"-s{population_size}_{num_clust} " # population size & number of clusters
                f"\"-ib_{bot_instance_0};c_{maxcut_instance_0}\" " # problem instance
                f"-f{approximation_front_file} " # front approximation
                "9 " # problem index
                "2 " # number of objectives
                f"{l} " # number of parameters
                "1000 " # elitist archive size target
                f"{number_of_evaluations} " # evaluation budget
                f"{number_of_evaluations}"  # logging interval (in #evaluations)
                " || true"
                "\n"))
    file_mode = 'a'
# Note: The fronts still have to reconstructed due to the decompostion.

# 6. BOT vs BOT
fnss = [1, 2, 4, 8, 16]

# Format: (n, (bot_instance_0, bot_instance_1))
ns_and_instances_bot_bot_6 = ns_and_instances_bot_4 = chain.from_iterable(
    zip(
        repeat(l),
        ((fns, f"./bestoftraps/bot_n{l}k5fns{fns}s0.txt", f"./bestoftraps/bot_n{l}k5fns{fns}s1.txt") for fns in fnss),
        
    )
    for l in ls
)

if run_bot_vs_bot:
    with open('experiments.txt', file_mode) as f:
        for a, run_idx, (l, (fns, bot_instance_0, bot_instance_1)) in product(approaches, range(number_of_runs), ns_and_instances_bot_bot_6):
            # Avoid giving small instances too many evaluations.
            number_of_evaluations = get_number_of_evaluations(l)
            approximation_front_file = f"./reference_fronts/bot_bot/n_{l}__fns_{fns}__k_5__s0_0__s1_1.txt"
            f.write((
                f"timeout {timeout_time} "
                "./build/MO_GOMEA "
                f"-a{a} " # approach
                f"-oresults/{exp_name}_bot_vs_bot/approach_{a.replace(' ', '-')}__run_{run_idx}__l_{l}__fns_{fns} " # output folder
                f"-s{population_size}_{num_clust} " # population size & number of clusters
                f"\"-if_{bot_instance_0};f_{bot_instance_1}\" " # problem instance
                f"-f{approximation_front_file} " # front approximation
                "7 " # problem index
                "2 " # number of objectives
                f"{l} " # number of parameters
                "1000 " # elitist archive size target
                f"{number_of_evaluations} " # evaluation budget
                f"{number_of_evaluations}"  # logging interval (in #evaluations)
                " || true"
                "\n"))
    file_mode = 'a'
# Note: The fronts still have to reconstructed due to the decompostion.