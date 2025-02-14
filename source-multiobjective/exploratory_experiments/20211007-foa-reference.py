#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
#
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
#
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

from itertools import product, chain, repeat
import math


approaches = [
    # Approaches in part 1: included for reference.
    # No extreme kernels
    # "-10_0 -c8 -x9_-3 -F0_1", # Original approach
    # Assign Extreme Kernels by KNN per niche
    # "-10_10_0.2 -c8 -x9_-3 -F0_1", # Original approach
    # "-10_10_0.2 -c8 -x9_-3 -F11_1", # Use scalarized for mixing with FI, draw random from archive
    # "-10_10_0.2 -c8 -x9_-3 -F11_1_0", # ^ + Single Objective FI uses the same strategy
    # "-10_10_0.2 -c8 -x9_-3 -F15_1", # Nearer Better (according to scalarization & hamming distance)
    # "-10_10_0.2 -c8 -x9_-3 -F15_1_0", # ^ + Single Objective FI uses the same strategy
    
    # Assign Extreme Kernels by subselection of KNN per niche
    # (in particular, cover each change in parameter)
    # "-10_11_0.2 -c8 -x9_-3 -F0_1",

    # Part 2
    # "-21_11_0.2 -c8 -x9_-3 -F15" # 'Favorite' configuration - S(ubset)D(ensity)UHV
    # "-19_11_0.2 -c8 -x9_-3 -F15" # S(ubset)UHV (i.e. no density correction)
    # "-22_11_0.2 -c8 -x9_-3 -F15" # D(ensity)UHV (i.e. no usage of subsets)

    # Part 3 - With fi mode 11 instead.
    # "-21_11_0.2 -c8 -x9_-3 -F11", # 'Favorite' configuration - S(ubset)D(ensity)UHV
    # "-19_11_0.2 -c8 -x9_-3 -F11", # S(ubset)UHV (i.e. no density correction)
    # "-22_11_0.2 -c8 -x9_-3 -F11", # D(ensity)UHV (i.e. no usage of subsets)

    # Reference approaches
    "-3 -c8 -x9_-3 -F0",
    "-3 -c8 -x9_-3 -F11",
    "-3 -c8 -x9_-3 -F15",

    # Not so much reference, but with local search
    "-3 -c8 -x9_-3 -F11 -l5",
    "-19_11_0.2 -c8 -x9_-3 -F11 -l5",
    "-21_11_0.2 -c8 -x9_-3 -F11 -l5",
    "-22_11_0.2 -c8 -x9_-3 -F11 -l5",
]

# Repeat a few times for good measure.
number_of_runs = 20
# Use population sizing scheme.
population_size = 2048
# Use a number of clusters equal to sqrt(|P|)
# Note that this does not affect results in general.
num_clust = "s5"
# 
ns = [100]
#
max_number_of_evaluations = 10_000_000
timeout_time = "10m"
search_space_multiplier = 8
capped_n = math.log2(max_number_of_evaluations / search_space_multiplier)

def get_number_of_evaluations(n: int):
    if n >= capped_n: return max_number_of_evaluations
    return 2**(n) * 4

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
        for a, run_idx, n in product(approaches, range(number_of_runs), ns):
            # Avoid giving small instances too many evaluations.
            number_of_evaluations = get_number_of_evaluations(n)
            f.write((
                f"timeout {timeout_time} "
                "./build/MO_GOMEA "
                f"-a{a} " # approach
                f"\"-oresults/foa_onemax_vs_zeromax/approach_{a.replace(' ', '-')}__run_{run_idx}__n_{n}\" " # output folder
                f"-s{population_size}_{num_clust} " # population size & number of clusters
                "0 " # problem index
                "2 " # number of objectives
                f"{n} " # number of parameters
                "1000 " # elitist archive size target
                f"{number_of_evaluations} " # evaluation budget
                f"{number_of_evaluations}"  # logging interval (in #evaluations)
                " || true"
                "\n"))
    file_mode = 'a'


# 2. MaxCut vs OneMax

# Format: (n, maxcut_instance_0)
ns_and_instances_maxcut_2 = (
    (n, f"./maxcut/maxcut_instance_{n}_0.txt")
    for n in ns
)

if run_maxcut_vs_onemax:
    with open('experiments.txt', file_mode) as f:
        for a, run_idx, (n, maxcut_instance_0) in product(approaches, range(number_of_runs), ns_and_instances_maxcut_2):
            # Avoid giving small instances too many evaluations.
            number_of_evaluations = get_number_of_evaluations(n)
            approximation_front_file = f"./reference_fronts/maxcut_onemax/n_{n}__i_0.txt"
            f.write((
                f"timeout {timeout_time} "
                "./build/MO_GOMEA "
                f"-a{a} " # approach
                f"-oresults/foa_maxcut_vs_onemax/approach_{a.replace(' ', '-')}__run_{run_idx}__n_{n} " # output folder
                f"-s{population_size}_{num_clust} " # population size & number of clusters
                f"-i{maxcut_instance_0} " # problem instance
                f"-f{approximation_front_file} " # front approximation
                "6 " # problem index
                "2 " # number of objectives
                f"{n} " # number of parameters
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
        for a, run_idx, n in product(approaches, range(number_of_runs), ns):
            # Avoid giving small instances too many evaluations.
            number_of_evaluations = get_number_of_evaluations(n)
            f.write((
                f"timeout {timeout_time} "
                "./build/MO_GOMEA "
                f"-a{a} " # approach
                f"-oresults/foa_maxcut_vs_maxcut/approach_{a.replace(' ', '-')}__run_{run_idx}__n_{n} " # output folder
                f"-s{population_size}_{num_clust} " # population size & number of clusters
                "4 " # problem index
                "2 " # number of objectives
                f"{n} " # number of parameters
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
    for n in ns
)
if run_bot_vs_onemax:
    with open('experiments.txt', file_mode) as f:
        for a, run_idx, (n, fns, bot_instance_0) in product(approaches, range(number_of_runs), ns_and_instances_bot_4):
            # Avoid giving small instances too many evaluations.
            number_of_evaluations = get_number_of_evaluations(n)
            approximation_front_file = f"./reference_fronts/bot_onemax/n_{n}__fns_{fns}__s_0__k_5.txt"
            f.write((
                f"timeout {timeout_time} "
                "./build/MO_GOMEA "
                f"-a{a} " # approach
                f"-oresults/foa_bot_vs_onemax/approach_{a.replace(' ', '-')}__run_{run_idx}__n_{n}__fns_{fns} " # output folder
                f"-s{population_size}_{num_clust} " # population size & number of clusters
                f"-if_{bot_instance_0} " # problem instance
                f"-f{approximation_front_file} " # front approximation
                "8 " # problem index
                "2 " # number of objectives
                f"{n} " # number of parameters
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
    for n in ns
)
if run_bot_vs_maxcut:
    with open('experiments.txt', file_mode) as f:
        for a, run_idx, (n, fns, bot_instance_0, maxcut_instance_0) in product(approaches, range(number_of_runs), ns_and_instances_bot_maxcut_5):
            # Avoid giving small instances too many evaluations.
            number_of_evaluations = get_number_of_evaluations(n)
            approximation_front_file = f"./reference_fronts/bot_maxcut/l_{n}__fns_{fns}__s_0__k_5.txt"
            f.write((
                f"timeout {timeout_time} "
                "./build/MO_GOMEA "
                f"-a{a} " # approach
                f"-oresults/foa_bot_vs_maxcut/approach_{a.replace(' ', '-')}__run_{run_idx}__n_{n}__fns_{fns} " # output folder
                f"-s{population_size}_{num_clust} " # population size & number of clusters
                f"\"-ib_{bot_instance_0};c_{maxcut_instance_0}\" " # problem instance
                f"-f{approximation_front_file} " # front approximation
                "9 " # problem index
                "2 " # number of objectives
                f"{n} " # number of parameters
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
        repeat(n),
        ((fns, f"./bestoftraps/bot_n{n}k5fns{fns}s0.txt", f"./bestoftraps/bot_n{n}k5fns{fns}s1.txt") for fns in fnss),
        
    )
    for n in ns
)

if run_bot_vs_bot:
    with open('experiments.txt', file_mode) as f:
        for a, run_idx, (n, (fns, bot_instance_0, bot_instance_1)) in product(approaches, range(number_of_runs), ns_and_instances_bot_bot_6):
            # Avoid giving small instances too many evaluations.
            number_of_evaluations = get_number_of_evaluations(n)
            approximation_front_file = f"./reference_fronts/bot_bot/n_{n}__fns_{fns}__k_5__s0_0__s1_1.txt"
            f.write((
                f"timeout {timeout_time} "
                "./build/MO_GOMEA "
                f"-a{a} " # approach
                f"-oresults/foa_bot_vs_bot/approach_{a.replace(' ', '-')}__run_{run_idx}__n_{n}__fns_{fns} " # output folder
                f"-s{population_size}_{num_clust} " # population size & number of clusters
                f"\"-if_{bot_instance_0};f_{bot_instance_1}\" " # problem instance
                f"-f{approximation_front_file} " # front approximation
                "7 " # problem index
                "2 " # number of objectives
                f"{n} " # number of parameters
                "1000 " # elitist archive size target
                f"{number_of_evaluations} " # evaluation budget
                f"{number_of_evaluations}"  # logging interval (in #evaluations)
                " || true"
                "\n"))
    file_mode = 'a'
# Note: The fronts still have to reconstructed due to the decompostion.