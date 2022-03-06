import numpy as np
from io import IOBase
from itertools import product
from pathlib import Path

def generate_random_nk_landscape(f: IOBase, L: int, k: int, Q: int, seed: int):
    """
    :param f: File to write to
    :param L: String length
    :param k: Subfunction size
    :param Q: Quantization (number of possible fitness value within subfunction)
    :param seed: Seed used in the rng
    """
    # We have 16 subfunctions!
    n_sfn = L
    rng = np.random.default_rng(seed=seed)
    # Variables per subfunction
    subfunction_variables = np.zeros((n_sfn, k), int)
    for sfn in range(n_sfn):
        subfunction_variables[sfn, 0] = sfn
        # Generate remaining without replacement, excluding sfn as a possibility
        # By making 0 impossible and replacing any occurence of sfn with 0.
        subfunction_variables[sfn, 1:] = rng.choice(range(L - 1), size=k-1, replace=False) + 1
        subfunction_variables[sfn, 1:][subfunction_variables[sfn, 1:] == sfn] = 0

    # Tabulated subfunctions
    subfunction_table = rng.choice(range(Q), size=(n_sfn, 2**k))

    ## Write start
    # Format of file is
    # <number of variables> <number of subfunctions>
    # --- For each subfunction:
    # <number of variables in subfunction # (v)> <listing of v variables>
    #    --- For each of the 2^v binary strings:
    # "<binary string>" <value>

    f.write(str(L))
    f.write(' ')
    f.write(str(n_sfn))
    f.write('\n')
    for sfn_variables, sfn_table in zip(subfunction_variables, subfunction_table):
        # Write variable spec
        f.write(str(len(sfn_variables)))
        f.write(' ')
        for v in sfn_variables:
            f.write(str(v))
            f.write(' ')
        f.write('\n')
        # Write function table
        for i, value in enumerate(sfn_table):
            f.write('"')
            ib = bin(i)[2:]
            ib = '0' * (k - len(ib)) + ib
            f.write(ib) # Write index as binary string, removing the starting 0b
            f.write('"')
            f.write(' ')
            f.write(str(value))
            f.write('\n')

Ls = (50, 100)
ks = (5,)
Qs = (16, 32, 64)
seeds = (42, 43, 44)

for L, k, Q, seed in product(Ls, ks, Qs, seeds):

    directory = Path("./instances_random") / f"L_{L}" / f"k_{k}" / f"Q_{Q}"
    directory.mkdir(parents=True, exist_ok=True)
    file = directory / f"seed_{seed}.txt"
    
    with file.open('w') as f:
        generate_random_nk_landscape(f, L=L, k=k, Q=Q, seed=seed)