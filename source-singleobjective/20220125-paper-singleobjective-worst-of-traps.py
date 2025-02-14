#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
#
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
#
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

from itertools import product, chain, repeat
import subprocess

import math

# Get git hash
# from https://stackoverflow.com/a/21901260/4224646
def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

exp_name = "paper_singleobjective_" + get_git_revision_short_hash()

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
run_worst_of_maxcut = True
file_mode = 'w'

# 2. MaxCut

# Format: (n, maxcut_instance)
def get_ns_and_instances_worst_of_maxcut(subfolder=False):
    maxcut_path = "./data/maxcut"
    if subfolder:
        maxcut_path = "../data/maxcut"
    return (
        (l, f"\"{maxcut_path}/maxcut_instance_{l}_0.txt;{maxcut_path}/maxcut_instance_{l}_1.txt\"", f"./data/wo_maxcut/wo_maxcut_{l}__0_1.vtr")
        for l in ls_maxcut
    )

if run_worst_of_maxcut:
    with open('experiments.txt', file_mode) as f:
        # GOMEA
        for (human_a, a), run_idx, (l, maxcut_instance, vtr_path) in product(gomea_approaches, range(run_start_index, run_start_index + number_of_runs), get_ns_and_instances_worst_of_maxcut()):
            # Avoid giving small instances too many evaluations.
            number_of_evaluations = get_number_of_evaluations(l)
            vtr = -1.0
            with open(vtr_path) as vtrf:
                vtr = float(vtrf.readline())
            f.write((
                f"timeout {timeout_time} "
                f"{gomea_path} "
                f"--approach={a} " # approach
                f"--folder=results/{exp_name}_maxcut/approach_{human_a.replace(' ', '-')}__run_{run_idx}__l_{l} " # output folder
                f"--scheme=0 " # population size & number of clusters
                "--problem=17 " # problem index
                "--alphabet=2 "
                f"--L={l} " # number of parameters
                f"--instance={maxcut_instance} " # problem instance
                f"--vtr={vtr} " # value to reach
                f"--seed={get_seed(run_idx)} "
                f"--maxEvals={number_of_evaluations} " # evaluation budget
                f"--timeLimit={time_limit} "
                " || true"
                "\n"))
        # DSMGA-II with IMS
        for run_idx, (l, maxcut_instance, vtr_path) in product(range(run_start_index, run_start_index + number_of_runs), get_ns_and_instances_worst_of_maxcut(subfolder = True)):
            #
            number_of_evaluations = get_number_of_evaluations(l)
            vtr = -1.0
            with open(vtr_path) as vtrf:
                vtr = float(vtrf.readline())
            #
            initial_population_size = 4
            maximum_number_of_generations = 200 # Same as GOMEA
            f.write((
                f"cd {dsmga2_directory} && "
                f"timeout {timeout_time} "
                f"{dsmga2_path} "
                f"{l} {initial_population_size} "
                "11 " # problem
                f"{maximum_number_of_generations} {number_of_evaluations} {time_limit} "
                f"{vtr} "
                f"{get_seed(run_idx)} "
                f"../results/{exp_name}_maxcut/approach_DSMGA2__run_{run_idx}__l_{l} " # folder
                f"{maxcut_instance} " # instance
                " || true"
                "\n"
            ))
            
    file_mode = 'a'
