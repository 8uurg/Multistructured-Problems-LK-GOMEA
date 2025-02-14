#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
#
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
#
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

import argparse
import pathlib
import re

def existing_path(astr):
    p = pathlib.Path(astr)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"{astr} does not exist")
    return p

argparser = argparse.ArgumentParser(description="Decompose a BestOfTraps problem into subfunctions")
argparser.add_argument("file", type=existing_path)
args = argparser.parse_args()

fns_regex = re.compile("fns[0-9]+")

# Modify the filename according to a pattern
def process_filename(path: pathlib.Path, fnidx: int):
    filename = path.stem
    filename = filename.replace("bot", "botd") # indicate decomposed
    filename = re.sub(fns_regex, "", filename) # remove function count (decomposed implies fns = 1)
    filename = filename + f"fn{fnidx}{path.suffix}" # append function index
    return path.with_name(filename)

with open(args.file, 'r') as f_r:
    # Read function count
    fns = f_r.readline()
    idx = 0
    while True:
        # Read subfunction
        count_header = f_r.readline()
        if count_header == "": break
        optimum = f_r.readline()
        if optimum == "": break
        permutation = f_r.readline()
        if permutation == "": break
        outpath = process_filename(args.file, idx)
        # Export file
        with open(outpath, 'w') as f_w:
            f_w.write("1\n")
            f_w.write(count_header)
            f_w.write(optimum)
            f_w.write(permutation)
        # Increment counter
        idx += 1

