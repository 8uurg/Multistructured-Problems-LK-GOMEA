# This script can be used to run GOMEA in increasing population size.
# It will double the population size until all `n` runs hit the optimum, or when it hits the maximum population size.
# The scheme (=3), population size, directory and seed are provided by this script. 
# Other arguments passed to this script after a `--` will be passed on to GOMEA.

# TODO: Write this script.

import subprocess
import argparse
import sys
from threading import Thread
from pathlib import Path
import queue
from io import IOBase
from datetime import datetime
import time
import shutil

# if platform.system() == "Windows":
#     raise RuntimeError("Windows does not support select, and as such this procedure does not work on windows.")

# Pass on the arguments after the first exact `--` (without arguments)
idx = 0
while sys.argv[idx] != "--" and idx < len(sys.argv):
    idx += 1
gomea_args = sys.argv[idx+1:]

argparser = argparse.ArgumentParser(description="Sequential doubling population size search.")
argparser.add_argument("--base-folder", type=Path, required=True)
argparser.add_argument("--num", type=int, required=True)
argparser.add_argument("--base", type=int, default=16)
argparser.add_argument("--max", type=int, default=4096)
argparser.add_argument("--timeout", type=int, default=None)
args = argparser.parse_args(sys.argv[1:idx])

# How many GOMEAs to run in parallel.
num_parallel = args.num # 2

base_population = args.base
max_population = args.max

max_time = args.timeout

current_population_size = base_population

# Perform compilation -- disabled as the experiment runner already
# does this.
# - Create build directory
# subprocess.call(["meson", "build", "--buildtype=debugoptimized"])
# - Compile
# subprocess.call(["meson", "compile", "-C", "build"])

running = []
is_running = [True for _ in range(num_parallel)]
is_successful = [False for _ in range(num_parallel)]

def any_running():
    for i in range(num_parallel):
        if is_running[i]:
            return True
    return False

def any_unsuccessful():
    for i in range(num_parallel):
        if not is_successful[i]:
            return True
    return False


def watch_gomea_for_abort(i: int, process):
    ret = process.wait()
    is_running[i] = False


def watch_elitists(i: int, process):
    while is_running[i]:
        line = process.stdout.readline()
        line_s = line.decode('utf8').split()
        if len(line) < 5:
            # print(f"Skipped for {i}: {line}")
            if len(line) == 0:
                time.sleep(1)
            continue
        try:
            sevals = int(line_s[0])
            stime = float(line_s[1])
            sfitness = float(line_s[2])
            siskey = int(line_s[3])
            ssolution = line_s[4]
            # print(f"Run {i}: {line}")
            if siskey:
                print(f"Run {i}: hit!")
                is_successful[i] = True

        except ValueError:
            pass # 

while any_unsuccessful() and current_population_size <= max_population:
    for idx in range(num_parallel):
        # Note: we override the seed by 42 + idx.
        # Folder is set to ./result/parallel
        directory = args.base_folder.absolute()
        directory.mkdir(parents=True, exist_ok=True)
        seed = idx + 42
        experiment_dir = directory.with_name(f"{directory.parts[-1]}__seed{seed}__pop{current_population_size}")
        if experiment_dir.exists():
            shutil.rmtree(experiment_dir)

        # Start GOMEA
        g = []
        if max_time is not None:
            g = [f'timeout', '-k', str(max_time*2), str(max_time)]

        command = g + ['./build/GOMEA'] + gomea_args + [f"--seed={seed}", "--scheme=3", f"--populationSize={current_population_size}", f"--folder={experiment_dir}"]
        # print(command)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        is_running[idx] = True
        th = Thread(target=watch_gomea_for_abort, args=(idx, process))
        th.daemon = True
        th.start()
        running.append(process)
        # Start watching elitists file for changes.
        # print(f"Watching {experiment_dir / 'elitists.txt'}")
        watcher = subprocess.Popen(['tail', '-F', experiment_dir / 'elitists.txt'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        running.append(watcher)
        # Run a thread processing these new lines.
        th = Thread(target=watch_elitists, args=(idx, watcher))
        th.daemon = True
        th.start()

    while True:
        time.sleep(1)
        if not any_running():
            break
    time.sleep(1)      

    # Stop everything!
    for r in running:
        r.terminate()

    # Wait for 5 seconds...
    time.sleep(5)

    # And kill if they haven't exited.
    for r in running:
        try:
            r.kill()
        except:
            pass
    
    current_population_size *= 2