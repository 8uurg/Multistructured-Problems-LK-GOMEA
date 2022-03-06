approaches = [-1]
number_of_runs = 10

with open('experiments.txt', 'w') as f:
    for a in approaches:
        for run_idx in range(number_of_runs):
            f.write(f"./build/MO_GOMEA -a{a}_-1 -oresults/hvtest/hv__approach_{a}__run_{run_idx}__pop_512__clust_3 -s512_3 4 2 100 1000 1000000 10000\n")