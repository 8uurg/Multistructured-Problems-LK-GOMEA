# Generates fully connected maxcut instances,
# like the ones provided with the multiobjective code.

import numpy as np
import argparse
import itertools
from pathlib import Path

argparser = argparse.ArgumentParser()
argparser.add_argument("l", type=int)
argparser.add_argument("--seed", default=42, type=int)

parsed = argparser.parse_args()

l = parsed.l
seed = parsed.seed
out_path = Path(f"./data/maxcut/maxcut_instance_{l}_{seed}.txt")
rng = np.random.default_rng(seed=seed)
num_edges = (l * (l - 1)) // 2

with out_path.open('w') as f:
    f.write(f"{l} {num_edges}\n")
    for (a, b) in itertools.combinations(range(l), 2):
        w = rng.integers(low=1, high=5, endpoint=True)
        # Indices for maxcut start at 1, not 0.
        f.write(f"{a + 1} {b + 1} {w}\n")