#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
#
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
#
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

approaches = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]
number_of_runs = 10

with open('experiments.txt', 'w') as f:
    for a in approaches:
        for run_idx in range(number_of_runs):
            f.write(f"./build/MO_GOMEA -a{a}_-1 -oresults/hvtest/hv__approach_{a}__run_{run_idx}__pop_512__clust_3 -s512_3 4 2 100 1000 1000000 10000\n")