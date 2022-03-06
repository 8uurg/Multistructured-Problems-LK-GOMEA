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
    gzpath = (main_folder / f"{folder.name}_statistics").with_suffix('.csv.gz')
    
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
    statisticsfilepath = folder / 'statistics.dat'

    # Skip over directories without an elitists file.
    # Something must've gone wrong... Maybe an early OOM killer?
    if not statisticsfilepath.exists():
        print(f"Couldn't open {statisticsfilepath}. File not found.")
        return

    elitistfile = statisticsfilepath.open('r')

    # Skip header
    original_header = elitistfile.readline()
    
    # Write a header if first
    if is_first:
        csvfile.write(original_header.rstrip())
        for k in params.keys():
            csvfile.write(",")
            csvfile.write(str(k))
        csvfile.write("\n")

    # We'll be writing this string often, so preparing it ahead of time
    # seems like a good plan
    params_str = ',' + (','.join(params.values())) + '\n'

    line = elitistfile.readline().rstrip()
    while line != '':
        csvfile.write(line.replace(' ', ','))
        csvfile.write(params_str)
        line = elitistfile.readline().rstrip()
    
    # Done!

# name_attr_split_re = re.compile("^([A-Za-z]+)_([0-9A-Za-z\-]+)$")
def split_resultsfolder(folder):
    res = {}

    segments = folder.name.split("__")

    for segment in segments:
        # match = name_attr_split_re.match(segment)
        subsegments = segment.split("_", 1)
        # if match is None:
        if len(subsegments) <= 1:
            # Skip invalid segments
            continue
            # raise RuntimeError(f"Failure Matching for segment '{segment}'")
        # res[match.group(1)] = match.group(2)
        res[subsegments[0]] = subsegments[1]

    return res


sub_folders = main_folder.glob("*")
for sub_folder in sub_folders:
    process_subfolder(sub_folder)
