#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
#
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
#
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

from itertools import product, chain, repeat
import math

# In order to compute a normalized value for hypervolume,
# we need to have a reference (or target) front.
# As the fronts are problem specific, 
# the problem instances need to be known ahead of time.

# The problems we wish to combine are as follows
# - OneMax (/ZeroMax), while simple has interesting behavior as good solutions
#   are sparse. Parameterized solely by the number of parameters (L). Single objective optimum known.
# - MaxCut, standard problem that GOMEA is known to solve well in a single-objective
#   setting. Parameterized by instance.
# - Best-of-Traps, combination of Trap functions that is hard to solve with the standard
#   GOMEA approach due to mismatched structure in the subfunctions.
#   Parameterized by number_of_parameters (L), block_size (k), number of functions (fns)
#   As well as their random seed (instance based)
#
# For all problems L = [6, 12, 25, 50, 100]
# - With this all problems for OneMax are fully specified.
# - These sizes were chosen as they match up with the maxcut instances already provided.
#   We will use the same instances that are already mapped out for maxcut as this
#   saves the work for determining the fronts belonging to these problems.
#   (Note: We may want to include some grid instances at a later point)
# - For best of traps we have various possible parameterizations.
#   We fix k to a singular value. Making this parameter too large makes this problem theoretically
#   hard to solve as the probability of an optimal block becomes too low,
#   and we end up with a needle-in-a-haystack problem. While too small makes the problem
#   too easy to be of interest. As this problem is derivative of the trap function we use
#   the same default: k = 5.
# - From prior experiments we know that fns > 16 in many cases too much. Even
#   for specifically adjusted approaches in a single objective setting. This however also makes
#   it difficult. As such fns in [1, 2, 4, 8, 16].
#   We will pregenerate the instances as C++'s <random> standard library may have different number generators across compilers
#   (it is not portable), in addition to this being required for employing the method below.
# - As problems with fns > 1 are hard to solve we employ a particular property of Best-of-Traps
#   to obtain high quality approximation fronts. 
#   Best-of-Traps takes the maximum value over its subfunctions. This matches up nicely with the
#   domination criterion, in which the higher point dominates the smaller points.
#   This match up allows us to do the following decomposition:
#   
#   Given a problem Q and B, where B is a Best-of-Traps problem consisting of {B_0, ..., B_{fns}}
#   as subproblems (permuted traps). Let P be a function that provides the pareto front to the problem.
#   And R a function that prunes dominated points from a set.
#   Then given P(Q, B_0), ... P(Q, B_{fns}), one can show that
#   P(Q, B) = R(U_{i=0}^fns P(Q, B_i))
#   
#   As P(Q, B_i) is simpler in structure than P(Q, B), heuristic approaches such as GOMEA
#   are more likely to be successful in providing a high quality reference front this way.
#   Furthermore, instances with lower fns consist of the same functions as lower ones.
# !: While this saves time, this does mean that the instances are correlated. This should be discussed.

# Summarizing: we need fronts for the following experiments
# ~1.~ OneMax vs ZeroMax (known: closed form)
#  2.  MaxCut vs OneMax
# ~3.~ MaxCut vs MaxCut (known: provided already)
#  4.  BOT vs OneMax -- using BOT \w decomposition (instance s0)
#  5.  BOT vs MaxCut -- using BOT \w decomposition (instance s0)
#  6.  BOT vs BOT    -- using 2x BOT \w decomposition (instance s0 and s1)

# HV should preferably be <= 1. 
# As letting each approach that we'll be evaluating run for an extended amount of time will
# provide this guarantee somewhat. The approaches to use are determined by the approaches we want to compare in the end
# 
# These are as follows
# -1 -- Tschebysheff Scalarized MO-GOMEA (Reference Approach using scalarizations)
#   Scalarized approaches with reference directions perform well and are fast and are common in literature.
#  0 -- MO-GOMEA (Original)
#   We should compare the original approach, as having a notion of "where we came from" is exceedingly useful
#   to determine whether an altercation provides a performance improvement.
#  7 -- Kernel MO-GOMEA (Domination/Kernel-Line, KNN (Line, Objective Space))
# 11 -- Kernel MO-GOMEA (Domination/Kernel-Line, KNN (Line) with 2k + Reverse Hamming Distance (k))
# 14 -- Linear Scalarized MO-GOMEA (Domination/Assigned-Line, Cluster Samples)
#   These approaches/configurations were ranked first, second and third respectively.
approaches = ["-1", "0", "7_-1", "11_-1"]
# Removed: "14_-1" -- too slow compared to the other approaches. Takes >30minutes where others take at most 2.
# Repeat a few times for good measure.
number_of_runs = 3
# Use population sizing scheme.
population_size = -1
# Go with a fixed number of 5 clusters.
num_clust = 5
# 
ns = [6, 12, 25, 50, 100]
#
max_number_of_evaluations = 10_000_000
search_space_multiplier = 4
capped_n = math.log2(max_number_of_evaluations / search_space_multiplier)

def get_number_of_evaluations(n: int):
    if n >= capped_n: return max_number_of_evaluations
    return 2**(n) * 4

# Flags for enabling / disabling particular subexperiments.
run_2_maxcut_vs_onemax = True
run_4_bot_vs_onemax = True
run_5_bot_vs_maxcut = True
run_6_bot_vs_bot = True

file_mode = 'w'

# 2. MaxCut vs OneMax

# Format: (n, maxcut_instance_0)
ns_and_instances_maxcut_2 = (
    (n, f"./maxcut/maxcut_instance_{n}_0.txt")
    for n in ns
)

if run_2_maxcut_vs_onemax:
    with open('experiments.txt', file_mode) as f:
        for a, run_idx, (n, maxcut_instance_0) in product(approaches, range(number_of_runs), ns_and_instances_maxcut_2):
            # Avoid giving small instances too many evaluations.
            number_of_evaluations = get_number_of_evaluations(n)
            f.write(f"./build/MO_GOMEA -a{a} -oresults/approxfront_maxcut_onemax/approach_{a}__run_{run_idx}__n_{n} -i{maxcut_instance_0} -s-1_{num_clust} 6 2 {n} 1000 {number_of_evaluations} {number_of_evaluations}\n")
    file_mode = 'a'
# 4. BOT vs OneMax
fns = range(16)
# Format: (n, bot_instance_0)
ns_and_instances_bot_4 = chain.from_iterable(
    ((n, fn, f"./bestoftraps/botd_n{n}k5s0fn{fn}.txt") for fn in fns) 
    for n in ns
)
if run_4_bot_vs_onemax:
    with open('experiments.txt', file_mode) as f:
        for a, run_idx, (n, fn, bot_instance_0) in product(approaches, range(number_of_runs), ns_and_instances_bot_4):
            # Avoid giving small instances too many evaluations.
            number_of_evaluations = get_number_of_evaluations(n)
            f.write(f"./build/MO_GOMEA -a{a} -oresults/approxfront_bot_onemax/approach_{a}__run_{run_idx}__n_{n}__fn_{fn} -if_{bot_instance_0} -s-1_{num_clust} 8 2 {n} 1000 {number_of_evaluations} {number_of_evaluations}\n")
    file_mode = 'a'
# Note: The fronts still have to reconstructed due to the decompostion.

# 5. BOT vs MaxCut
fns = range(16)
# Format: (n, bot_instance_0, maxcut_instance_0)
ns_and_instances_bot_maxcut_5 = ns_and_instances_bot_4 = chain.from_iterable(
    (((n, fn, f"./bestoftraps/botd_n{n}k5s0fn{fn}.txt", f"./maxcut/maxcut_instance_{n}_0.txt") for fn in fns)) 
    for n in ns
)
if run_5_bot_vs_maxcut:
    with open('experiments.txt', file_mode) as f:
        for a, run_idx, (n, fn, bot_instance_0, maxcut_instance_0) in product(approaches, range(number_of_runs), ns_and_instances_bot_maxcut_5):
            # Avoid giving small instances too many evaluations.
            number_of_evaluations = get_number_of_evaluations(n)
            f.write(f"./build/MO_GOMEA -a{a} -oresults/approxfront_bot_maxcut/approach_{a}__run_{run_idx}__n_{n}__fn_{fn} \"-ib_{bot_instance_0};c_{maxcut_instance_0}\" -s-1_{num_clust} 9 2 {n} 1000 {number_of_evaluations} {number_of_evaluations}\n")
    file_mode = 'a'
# Note: The fronts still have to reconstructed due to the decompostion.

# 6. BOT vs BOT
fns_0 = range(16)
fns_1 = range(16)

# Lower the number of runs for this one, we already have multiple approaches running.
# The quadratic growth already makes this one quite large.
number_of_runs = 1

# Format: (n, (bot_instance_0, bot_instance_1))
ns_and_instances_bot_bot_6 = ns_and_instances_bot_4 = chain.from_iterable(
    zip(
        repeat(n),
        product(
            ((fn, f"./bestoftraps/botd_n{n}k5s0fn{fn}.txt") for fn in fns_0),
            ((fn, f"./bestoftraps/botd_n{n}k5s1fn{fn}.txt") for fn in fns_1)
        )
    )
    for n in ns
)

if run_6_bot_vs_bot:
    with open('experiments.txt', file_mode) as f:
        for a, run_idx, (n, ((fn_0, bot_instance_0), (fn_1, bot_instance_1))) in product(approaches, range(number_of_runs), ns_and_instances_bot_bot_6):
            # Avoid giving small instances too many evaluations.
            number_of_evaluations = get_number_of_evaluations(n)
            f.write(f"./build/MO_GOMEA -a{a} -oresults/approxfront_bot_bot/approach_{a}__run_{run_idx}__n_{n}__fna_{fn_0}__fnb_{fn_1} \"-if_{bot_instance_0};f_{bot_instance_1}\" -s-1_{num_clust} 7 2 {n} 1000 {number_of_evaluations} {number_of_evaluations}\n")
    file_mode = 'a'
# Note: The fronts still have to reconstructed due to the decompostion.