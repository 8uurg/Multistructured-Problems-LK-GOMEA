# This file contains a modified version of `merge_and_filter_fronts.py`. It merges
# the fronts of individual Best-Of-Traps functions back together in a single batch.

import argparse
import pathlib
from itertools import chain
import pandas as pd
import numpy as np
from typing import List

def merge_front_files(in_files : List[pathlib.Path], out_file: pathlib.Path, instance=""):
    is_instance_empty = True
    instance_origin = None

    for in_file in in_files:
        in_file_instance = ""
        with open(in_file, 'r', newline='', encoding='utf-8') as f:
            f.readline()
            in_file_instance = f.readline().strip()
        if is_instance_empty:
            instance = in_file_instance
            is_instance_empty = instance == ""
            instance_origin = in_file
        else:
            assert instance == in_file_instance, f"Instance specifiers of fronts to merge do not match:\nFile `{instance_origin}` has `{instance}` while file `{in_file}` has `{in_file_instance}`"


    # Combine arrays together
    concatenated = pd.concat([pd.read_csv(in_file, skiprows=2, header=None, sep=" ") for in_file in in_files])

    # Filter using brute force.
    array = concatenated.to_numpy().reshape(concatenated.shape[0], 1, concatenated.shape[1])
    A = array
    B = np.swapaxes(array, 0, 1)
    Dstrict = np.max(A < B, axis=2)
    Dloose = np.min(A <= B, axis=2)
    D = Dstrict & Dloose # for each [i, j]: is i dominated by j?
    E = np.tril(Dloose & Dloose.T, k=-1) # for each [i, j]: is i equal to j? (after tril: and i > j)

    filtered = concatenated[~np.max(D, axis=1) & ~np.max(E, axis=1)] # filter out solutions that are dominated

    # Sort (in part for nicer viewing, but also for merging & tracking)
    filtered = filtered.sort_values(list(range(filtered.shape[1])), ascending=False)

    with open(out_file, 'w', newline='', encoding='utf-8') as f_w:

        number_of_objectives = filtered.shape[1]
        number_of_points = filtered.shape[0]
        # Write #objectives and #points
        f_w.write(f"{number_of_objectives} {number_of_points}\n")
        # Write instance specifier ()
        f_w.write(f"{instance}\n")
        # Write points
        filtered.to_csv(f_w, sep=" ", header=False, index=False)
        # Done!

def path_that_is_a_folder(astr):
    p = pathlib.Path(astr)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"{astr} does not exist")
    if p.is_file():
        raise argparse.ArgumentTypeError(f"{astr} is not a folder")
    return p

def path_fresh_folder(astr):
    p = pathlib.Path(astr)
    if p.exists():
        raise argparse.ArgumentTypeError(f"{astr} exists already")
    p.mkdir()
    return p

argparser = argparse.ArgumentParser()
argparser.add_argument("folder_in", type=path_that_is_a_folder)
argparser.add_argument("folder_out", type=path_fresh_folder)
args = argparser.parse_args()

# Filename parsing
def process_file(file: pathlib.Path):
    attribute_value_pairs = dict(map(lambda s: s.split("_", 1), file.stem.split("__")))
    return {
        "file": file,
        **attribute_value_pairs
    }

# Merge all runs.
directory_in = pathlib.Path(args.folder_in)
files = pd.DataFrame([process_file(file) for file in directory_in.glob("*.txt")])
# Ensure int
files["fna"] = files["fna"].astype(int)
files["fnb"] = files["fnb"].astype(int)
# Construct mapping for fns <-> fn
# for BOT vs BOT we only view the case of equal amount of subfunctions
instance_combinations = pd.DataFrame(chain.from_iterable(
    ({"fns": fns, "fna": fna, "fnb": fnb} for fna in range(fns) for fnb in range(fns))
    for fns in [1, 2, 4, 8, 16]
))
files = pd.merge(files, instance_combinations, on=["fna", "fnb"])

# Columns to merge.
columns_to_merge = ["file", "approach", "run", "fna", "fnb"]
columns_to_group_by = [c for c in files.columns if c not in columns_to_merge]
directory_out = pathlib.Path(args.folder_out)

def perform_merger(df: pd.DataFrame):
    attributes = df.iloc[0][columns_to_group_by]
    stem = "__".join(f"{a}_{attributes[a]}" for a in columns_to_group_by)
    # EDIT: Extra attributes
    filename = f"{stem}__k_5__s0_0__s1_1.txt"
    files_in = list(df["file"])
    file_out = directory_out / filename
    # EDIT: Provide instance
    n = attributes["n"]
    fns = attributes["fns"]
    bot_instance_0 = f"./bestoftraps/bot_n{n}k5s0fns{fns}.txt"
    bot_instance_1 = f"./bestoftraps/bot_n{n}k5s1fns{fns}.txt"
    instance = f"f_{bot_instance_0};f_{bot_instance_1}"
    # print(f"merge {files_in} to into {file_out}. Instance {instance}")
    merge_front_files(files_in, file_out, instance=instance)

files.groupby(columns_to_group_by).apply(perform_merger)