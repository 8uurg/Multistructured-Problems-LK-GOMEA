import argparse
import pathlib
import pandas as pd
import numpy as np

def existing_path(astr):
    p = pathlib.Path(astr)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"{astr} does not exist")
    return p

argparser = argparse.ArgumentParser(description="Convert archive dump to front file.")
argparser.add_argument("output", help="path to write the front file to", type=pathlib.Path)
argparser.add_argument("files", help="fronts to process", type=existing_path, nargs="+")
args = argparser.parse_args()

instance = ""
is_instance_empty = True
instance_origin = None

for in_file in args.files:
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
concatenated = pd.concat([pd.read_csv(in_file, skiprows=2, header=None, sep=" ") for in_file in args.files])

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

with open(args.output, 'w',  encoding='utf-8') as f_w:

    number_of_objectives = filtered.shape[1]
    number_of_points = filtered.shape[0]
    # Write #objectives and #points
    f_w.write(f"{number_of_objectives} {number_of_points}\n")
    # Write instance specifier ()
    f_w.write(f"{instance}\n")
    # Write points
    filtered.to_csv(f_w, sep=" ", header=False, index=False)
    # Done!