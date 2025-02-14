#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
#
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
#
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

from pathlib import Path
import re
# import subprocess
from tqdm import tqdm
import gzip

# Config
main_folder = Path("./results")

# This script turns results into a csv file, one per subfolder of
# {main_folder}, with each subfolder thereof containing parameters
# separated by a double underscore.

def process_subfolder(folder):
    # csvpath = (main_folder / folder.name).with_suffix('.csv')
    gzpath = (main_folder / folder.name).with_suffix('.he.csv.gz')
    
    # Skip over non-directories
    if not folder.is_dir():
        return

    # This folder has already been processed. Skip
    if gzpath.exists():
        return
    

    resultfolders = list(folder.glob("*"))
    is_first = True
    print('Aggregating...')
    csvfile = gzip.open(gzpath, 'wt')
    for resultfolder in tqdm(resultfolders):
        # Skip over non-directories (eg. metadata)
        if not resultfolder.is_dir():
            continue
        process_resultsfolder(resultfolder, csvfile, is_first)
        is_first = False
    csvfile.close()
    

def process_resultsfolder(folder, csvfile, is_first):
    params = split_resultsfolder(folder)
    hitevalfilepath = next(folder.glob("number_of_evaluations_when_all_points_foun*.dat"))

    # Skip over directories without an elitists file.
    # Something must've gone wrong... Maybe an early OOM killer?
    if not hitevalfilepath.exists():
        print(f"Couldn't open {hitevalfilepath}. File not found.")
        return

    hitevalfilefile = hitevalfilepath.open('r')

    # No header
    
    # Write a header if first
    if is_first:
        csvfile.write("#evaluations")
        for k in params.keys():
            csvfile.write(",")
            csvfile.write(str(k))
        csvfile.write("\n")

    # We'll be writing this string often, so preparing it ahead of time
    # seems like a good plan
    params_str = ',' + (','.join(params.values())) + '\n'

    line = hitevalfilefile.readline().rstrip()
    while line != '':
        csvfile.write(line.replace(' ', ','))
        csvfile.write(params_str)
        line = hitevalfilefile.readline().rstrip()
    
    # Done!

name_attr_split_re = re.compile("^([A-Za-z]+)([\-0-9]+|_[0-9A-Za-z\-]+|IMS|SPS|IMR)$")
def split_resultsfolder(folder):
    res = {}

    segments = folder.name.split("__")

    for segment in segments:
        match = name_attr_split_re.match(segment)
        if match is None:
            raise RuntimeError(f"Failure Matching for segment '{segment}'")

        r = match.group(2)
        if r.startswith("_"): r = r[1:]
        res[match.group(1)] = r

    return res


sub_folders = main_folder.glob("*")
for sub_folder in sub_folders:
    process_subfolder(sub_folder)
