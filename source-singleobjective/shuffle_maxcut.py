#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
#
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
#
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

# Generates fully connected maxcut instances,
# like the ones provided with the multiobjective code.

import numpy as np
import pandas as pd
import argparse
import itertools
from pathlib import Path

argparser = argparse.ArgumentParser()
argparser.add_argument("input", type=Path)
argparser.add_argument("--seed", default=42, type=int)

parsed = argparser.parse_args()


in_path : Path = parsed.input
seed = parsed.seed
out_path = in_path.with_suffix(f".s{seed}.txt")
print(f"Writing to {out_path}")

rng = np.random.default_rng(seed=seed)

with open(in_path) as in_file, open(out_path, 'w') as out_file:
    header = in_file.readline()
    header_s = header.split()
    l = int(header_s[0])
    # num_edges = int(header_s[1])
    out_file.write(header)

    remapping = rng.permutation(l)

    edges = pd.read_csv(in_file, sep=" ", header=None, names=["x", "y", "w"])
    edges["x"] = remapping[edges["x"] - 1] + 1
    edges["y"] = remapping[edges["y"] - 1] + 1
    edges.to_csv(out_file, sep=" ", index=False, header=None)

