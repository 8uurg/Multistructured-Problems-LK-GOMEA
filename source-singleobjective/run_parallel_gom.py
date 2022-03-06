# This script can be used to run GOMEA 30x in parallel (configure down below!)
# It will stop when all GOMEAs agree on the best solution (or when you press CTRL+C)
# The directory and seed are provided by this script. 
# Other arguments can be passed to this script, they will be passed on to GOMEA.

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

# How many GOMEAs to run in parallel.
num_parallel = 20
# Maximum time how long to wait after the last elitist update of any GOMEA.
timeout = 60 * 60 # 60 minutes without any update
timeout_agree = 20 # Once solutions agree. Set timeout to 20s instead.
# Note: it will be increased again upon an improvement

# Don't stop before we have spent atleast this many seconds!
minimum_time = 0 
# 5 * 60 # 5 minutes


# Perform compilation
# - Create build directory
subprocess.call(["meson", "build", "--buildtype=debugoptimized"])
# - Compile
subprocess.call(["meson", "compile", "-C", "build"])

# All arguments except for the python script name are all passed on to GOMEA
# We specifically define 
gomea_args = sys.argv[1:]

running = []

q = queue.Queue()

def watch_elitists(i: int, process):
    while True:
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
            q.put((i, sfitness, ssolution))
        except ValueError:
            pass # 

for idx in range(num_parallel):
    # Note: we override the seed by 42 + idx.
    # Folder is set to ./result/parallel
    directory = Path("./result/parallel")
    directory.mkdir(parents=True, exist_ok=True)
    experiment_dir = directory / f"{idx}"
    if experiment_dir.exists():
        shutil.rmtree(experiment_dir)

    # Start GOMEA
    command = ['./build/GOMEA'] + gomea_args + [f"--seed={42 + idx}", f"--folder={experiment_dir}"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    running.append(process)
    # Start watching elitists file for changes.
    print(f"Watching {experiment_dir / 'elitists.txt'}")
    watcher = subprocess.Popen(['tail', '-F', experiment_dir / 'elitists.txt'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    running.append(watcher)
    # Run a thread processing these new lines.
    th = Thread(target=watch_elitists, args=(idx, watcher))
    th.daemon = True
    th.start()


# Keep track!
best_fitness = 0.0
best_solution = ""
num_best_fitness = 0
current_best_fitness_idx = [0 for _ in range(num_parallel)]
current_best_solution_idx = ["" for _ in range(num_parallel)]
starting_time = datetime.now()
ctimeout = timeout

try:
    while True: 
        try:
            item = q.get(block=True, timeout=ctimeout)

            idx = item[0]
            fitness = item[1]
            solution = item[2]
            if current_best_fitness_idx[idx] >= fitness:
                # Should only happen if removal failed. Or a fitness is repeated (e.g. when searching for multiple
                # optima), this currently happens only with a known value, which this script is not meant for!
                # Nevertheless. This should avoid issues double-counting.
                continue
            current_best_fitness_idx[idx] = fitness

            current_best_solution_idx[idx] = solution

            ctimeout = timeout
            if fitness > best_fitness:
                best_fitness = fitness
                num_best_fitness = 1
                best_solution = solution
                print(f"Best across all improved ({idx}): {best_fitness}")
            elif fitness == best_fitness:
                num_best_fitness += 1
                # Note: can also check if they all found the same solution.
                # This should be fine...
                print(f"Best so far ({best_fitness}) is now joined by another solution {idx}. {num_best_fitness} total!")
            else:
                pass
                # print(f"{idx} had an improvement.")
            diff = datetime.now() - starting_time
            if num_best_fitness >= num_parallel:
                print("All GOMEA runs agree on a single fitness value...")
                ctimeout = timeout_agree
            # if diff.seconds > minimum_time and num_best_fitness == num_parallel:
            #    break

        except queue.Empty:
            # Hit timeout!
            diff = datetime.now() - starting_time
            print(f"Timeout hit!")
            if diff.seconds > minimum_time:
                break
    
except KeyboardInterrupt:
    print(f"Stopped via Ctrl+C!")

completion_time = datetime.now()
time_difference = completion_time - starting_time

print(f"Done! After {time_difference.total_seconds()}s best is {best_fitness} via solution [{best_solution}]")

# Done!
with open('best.txt', 'w') as f:
    f.write(str(best_fitness))
    f.write(' ')
    f.write(str(best_solution))

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
