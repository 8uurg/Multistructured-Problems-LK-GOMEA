#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
#
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
#
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

from itertools import product, chain, repeat
import math

# This experiment is the benchmark which `20210901-approximate_front.py` refers to.
#
# This experiment is performed on the following problems `<idx> - <name>`:
# 0 - OneMax vs ZeroMax (using built-in reference front, loads automatically)
# 6 - MaxCut vs OneMax (front obtained via `20210901-approximate_front.py`)
# 4 - MaxCut vs MaxCut (front was included with MO-GOMEA source code, loads automatically)
# 8 - Best-of-Traps vs OneMax (front obtained via `20210901-approximate_front.py`)
# 9 - Best-of-Traps vs MaxCut (front obtained via `20210901-approximate_front.py`)
# 7 - Best-of-Traps vs Best-of-Traps (front obtained via `20210901-approximate_front.py`)

# With the following approaches
# `-1` - Tschebysheff Scalarized MO-GOMEA
#           Luong, Ngoc Hoang, Han La Poutré, and Peter A. N. Bosman. 2018.
#           ‘Multi-Objective Gene-Pool Optimal Mixing Evolutionary Algorithm with the Interleaved Multi-Start Scheme’.
#           Swarm and Evolutionary Computation 40 (June): 238–54.
#           https://doi.org/10.1016/j.swevo.2018.02.005.
#
#  `0` - Domination-based MO-GOMEA - like 
#           Luong, Ngoc Hoang, Tanja Alderliesten, and Peter A. N. Bosman. 2018.
#           ‘Improving the Performance of MO-RV-GOMEA on Problems with Many Objectives Using Tchebycheff Scalarizations’.
#           In Proceedings of the Genetic and Evolutionary Computation Conference, 705–12. GECCO ’18. New York, NY, USA: Association for Computing Machinery.
#           https://doi.org/10.1145/3205455.3205498.
#
# -- Novel
#  `7` - Hybrid Linear Scalarization Kernel/Domination MO-GOMEA, objective space KNN neighborhood
# `11` - Hybrid Linear Scalarization Kernel/Domination MO-GOMEA, objective space KNN & parameter space KNN neighborhood
# `14` - Assigned Line MO-GOMEA, reverse KNN
approaches = ["-1", "0", "7_-1", "11_-1", "14_-1"]
# Note: "14_-1" is much slower than the other approaches.

# Repeat a few times for good measure.
number_of_runs = 20
# Use population sizing scheme.
population_size = -1
# Go with a fixed number of 5 clusters.
num_clust = 5
# 
ns = [6, 12, 25, 50, 100]
#
max_number_of_evaluations = 1_000_000
search_space_multiplier = 4
capped_n = math.log2(max_number_of_evaluations / search_space_multiplier)

def get_number_of_evaluations(n: int):
    if n >= capped_n: return max_number_of_evaluations
    return 2**(n) * 4

# Flags for enabling / disabling particular subexperiments.
run_onemax_vs_zeromax = False
run_maxcut_vs_onemax  = False
run_maxcut_vs_maxcut  = False
run_bot_vs_onemax     = True  # Rerun due to overlapping output directories.
run_bot_vs_maxcut     = False
run_bot_vs_bot        = True  # Rerun due to incorrect fronts used for computing hv.

file_mode = 'w'

# 1. Onemax vs Zeromax
if run_onemax_vs_zeromax:
    with open('experiments.txt', file_mode) as f:
        for a, run_idx, n in product(approaches, range(number_of_runs), ns):
            # Avoid giving small instances too many evaluations.
            number_of_evaluations = get_number_of_evaluations(n)
            f.write((
                "./build/MO_GOMEA "
                f"-a{a} " # approach
                f"-oresults/mo_benchmark_a_onemax_vs_zeromax/approach_{a}__run_{run_idx}__n_{n} " # output folder
                f"-s{population_size}_{num_clust} " # population size & number of clusters
                "0 " # problem index
                "2 " # number of objectives
                f"{n} " # number of parameters
                "1000 " # elitist archive size target
                f"{number_of_evaluations} " # evaluation budget
                f"{number_of_evaluations}"  # logging interval (in #evaluations)
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
                "./build/MO_GOMEA "
                f"-a{a} " # approach
                f"-oresults/mo_benchmark_a_maxcut_vs_onemax/approach_{a}__run_{run_idx}__n_{n} " # output folder
                f"-s{population_size}_{num_clust} " # population size & number of clusters
                f"-i{maxcut_instance_0} " # problem instance
                f"-f{approximation_front_file} " # front approximation
                "6 " # problem index
                "2 " # number of objectives
                f"{n} " # number of parameters
                "1000 " # elitist archive size target
                f"{number_of_evaluations} " # evaluation budget
                f"{number_of_evaluations}"  # logging interval (in #evaluations)
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
                "./build/MO_GOMEA "
                f"-a{a} " # approach
                f"-oresults/mo_benchmark_a_maxcut_vs_maxcut/approach_{a}__run_{run_idx}__n_{n} " # output folder
                f"-s{population_size}_{num_clust} " # population size & number of clusters
                "4 " # problem index
                "2 " # number of objectives
                f"{n} " # number of parameters
                "1000 " # elitist archive size target
                f"{number_of_evaluations} " # evaluation budget
                f"{number_of_evaluations}"  # logging interval (in #evaluations)
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
                "./build/MO_GOMEA "
                f"-a{a} " # approach
                f"-oresults/mo_benchmark_a_bot_vs_onemax/approach_{a}__run_{run_idx}__n_{n}__fns_{fns} " # output folder
                f"-s{population_size}_{num_clust} " # population size & number of clusters
                f"-if_{bot_instance_0} " # problem instance
                f"-f{approximation_front_file} " # front approximation
                "8 " # problem index
                "2 " # number of objectives
                f"{n} " # number of parameters
                "1000 " # elitist archive size target
                f"{number_of_evaluations} " # evaluation budget
                f"{number_of_evaluations}"  # logging interval (in #evaluations)
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
                "./build/MO_GOMEA "
                f"-a{a} " # approach
                f"-oresults/mo_benchmark_a_bot_vs_maxcut/approach_{a}__run_{run_idx}__n_{n}__fns_{fns} " # output folder
                f"-s{population_size}_{num_clust} " # population size & number of clusters
                f"\"-ib_{bot_instance_0};c_{maxcut_instance_0}\" " # problem instance
                f"-f{approximation_front_file} " # front approximation
                "9 " # problem index
                "2 " # number of objectives
                f"{n} " # number of parameters
                "1000 " # elitist archive size target
                f"{number_of_evaluations} " # evaluation budget
                f"{number_of_evaluations}"  # logging interval (in #evaluations)
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
                "./build/MO_GOMEA "
                f"-a{a} " # approach
                f"-oresults/mo_benchmark_a_bot_vs_bot/approach_{a}__run_{run_idx}__n_{n}__fns_{fns} " # output folder
                f"-s{population_size}_{num_clust} " # population size & number of clusters
                f"\"-if_{bot_instance_0};f_{bot_instance_1}\" " # problem instance
                f"-f{approximation_front_file} " # front approximation
                "7 " # problem index
                "2 " # number of objectives
                f"{n} " # number of parameters
                "1000 " # elitist archive size target
                f"{number_of_evaluations} " # evaluation budget
                f"{number_of_evaluations}"  # logging interval (in #evaluations)
                "\n"))
    file_mode = 'a'
# Note: The fronts still have to reconstructed due to the decompostion.