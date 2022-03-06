import argparse
import pathlib
import pandas as pd

def existing_path(astr):
    p = pathlib.Path(astr)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"{astr} does not exist")
    return p

argparser = argparse.ArgumentParser(description="Convert archive dump to front file.")
argparser.add_argument("file", help="archive dump to process", type=existing_path)
argparser.add_argument("output", help="path to write the front file to", type=pathlib.Path)
argparser.add_argument("--instance", help="specify the instance (such that the front can be checked to match the problem)", default="")
args = argparser.parse_args()

in_data = pd.read_csv(args.file).filter(regex="objective[0-9]+")

with open(args.output, 'w', newline='', encoding='utf-8') as f_w:
    number_of_objectives = in_data.shape[1]
    number_of_points = in_data.shape[0]
    # Write #objectives and #points
    f_w.write(f"{number_of_objectives} {number_of_points}\n")
    # Write instance specifier ()
    f_w.write(f"{args.instance}\n")
    # Write points
    in_data.to_csv(f_w, sep=" ", header=False, index=False)
    # Done!