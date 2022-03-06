import math
import subprocess
import pandas as pd
import numpy as np
import os
import argparse
from copy import deepcopy
import time
import shutil
from copy import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.metrics import r2_score
import math
from matplotlib import rcParams
rcParams['figure.dpi'] = 300

parser = argparse.ArgumentParser(description='parse parameters')

parser.add_argument('FIRST_RUN', metavar='FIRST_RUN', type=int,
                    help='number of optimization runs')
parser.add_argument('N_RUNS', metavar='N_RUNS', type=int,
                    help='number of optimization runs')
parser.add_argument('TIME_LIMIT', metavar='TIME_LIMIT', type=int,
                    help='number of optimization runs')
parser.add_argument('NUM_PROC', metavar='NUM_PROC', type=int,
                    default=8, help='number of optimization runs')

args_global = parser.parse_args()

PROBLEM_NAMES = {#1: 'Concatenated_Deceptive_Trap_4_4',
                 #2: 'Concatenated_Deceptive_Trap_4_3',
                 #3: 'Concatenated_Deceptive_Trap_4_1',
                 4: 'Concatenated_Deceptive_Trap_5_5',
                 #5: 'Concatenated_Deceptive_Trap_5_1',
                 6: 'Concatenated_Deceptive_Trap_5_4',                                  
                 #7: 'NK_4_1',
                 #8: 'NK_4_3',                 
                 9: 'NK_5_1',
                 #10: 'NK_6_1',                 
                 #11: 'MaxCut_Sparse',
                 #12: 'MaxCut_Dense',                 
                 13: 'Hierarhical_If-And-Only-If',
                 14: 'Bimodal_Concatenated_Deceptive_Trap_6_6',
                 15: 'Spinglass',
                 16: 'MAXSAT',
                 17: 'MaxCut_Dense2',
                 18: 'MaxCut_Sparse2'
                 }


DIMS = {
#1: [40, 80, 160, 320, 640, 1280],
#2: [42, 81, 162, 321, 642, 1284],
#3: [40, 80, 160, 320, 640, 1280],
4: [40, 80, 160, 320, 640, 1280],
#5: [40, 80, 160, 320, 640],
6: [40, 80, 160, 320, 640, 1280],
#7: [40, 80, 160, 320, 640, 1280],
#8: [40, 80, 160, 320, 640, 1280],
9: [40, 80, 160, 320, 640, 1280],
#10: [40, 80, 160, 320, 640],
#11: [49, 100, 196, 400, 784, 1600],
#12: [12, 25, 50, 100, 200],
13:  [64, 128, 256, 512, 1024, 2048],
14: [36, 78, 156, 318, 636, 1278],
15: [36, 100, 196, 400, 784],
16: [20, 50, 100, 200],
17: [12, 25, 50, 100, 200],
18: [49, 100, 196, 400, 784, 1600],
}

DIMS = {
16: [200],
}

def get_trap_vtr(L, k, s):
    #num_subfunctions = (L - k) // s + 1
    num_subfunctions = L // s
    vtr = num_subfunctions * k
    return vtr

def get_vtr_hiff(L):
    vtr = 0
    block_size = 1
    while block_size <= L:
        vtr += L
        block_size *= 2
    return vtr
       
def get_vtr_mixed_trap(L):
    vtr = 0
    i = 0
    num_functions = 0
    while i < L:
        if num_functions % 3 == 0:
            k = 4
        elif num_functions % 3 == 1:
            k = 6
        else:
            k = 8
        i += (k-1)
        vtr += k
        num_functions += 1
    return vtr

def get_vtr(problem_name, L, instance_number):

    if 'NK' in problem_name:     
        k = int(problem_name.split('_')[-2])
        s = int(problem_name.split('_')[-1])    
        vtr = np.loadtxt('../GOMEA_mod/problem_data/%s/L%d/%d_vtr.txt' % ('n%d_s%d' %(k,s), L, instance_number)) - 1e-6  

    elif problem_name == 'Spinglass':     
        vtr = open('SPIN/%d/%d_%d' % (L, L, instance_number), 'r').readlines()[1]
        vtr = -float(vtr)
        vtr -= 1e-6  


    elif 'Concatenated_Deceptive_Trap' in problem_name:
        k = int(problem_name.split('_')[-2])
        s = int(problem_name.split('_')[-1])
        vtr = get_trap_vtr(L, k, s)

    elif problem_name == 'MaxCut_Sparse':       
        vtr = np.loadtxt('../GOMEA_mod/problem_data/maxcut/set0b/n%.7di%.2d.bkv' % (L, instance_number))

    elif problem_name == 'MaxCut_Dense':       
        vtr = np.loadtxt('../GOMEA_mod/problem_data/maxcut/set0a/n%.7di%.2d.bkv' % (L, instance_number))

    elif problem_name == 'MaxCut_Dense2':       
        vtr = np.loadtxt('../GOMEA_mod/problem_data/maxcut/set0e/n%.7di%.2d.bkv' % (L, instance_number))

    elif problem_name == 'MaxCut_Sparse2':       
        vtr = np.loadtxt('../GOMEA_mod/problem_data/maxcut/set0c/n%.7di%.2d.bkv' % (L, instance_number))

    elif problem_name == 'Hierarhical_If-And-Only-If':
        vtr = get_vtr_hiff(L)

    elif problem_name == 'MixedTrap':
        vtr = get_vtr_mixed_trap(L)
    
    elif problem_name =='MAXSAT':
        vtr = 0.0

    else:
        print ('problem not implemented!')
        exit(0)

    return vtr

def get_data(folder_name):

    filename = os.path.join(folder_name, 'elitists.txt')
    if not os.path.exists(filename):
        return [],[]

    data = open(filename, 'r').readlines()
    if len(data) == 0:
        return [],[]

    best = data[-1].split(' ')
    
    n_evals = int(best[0])
    elitist_value = float(best[2])

    return n_evals, elitist_value
        
def solve_instance(num_runs, settings, problem_name, dim):

    n_evals_array, elitist_value_array = [], []

    for r in range(args_global.FIRST_RUN, num_runs):
        
        vtr = get_vtr(problem_name, dim, r+1)
        print(vtr)
        
        instance_name = '_'
        if 'Concatenated_Deceptive_Trap' in problem_name:
            k = str(problem_name.split('_')[-2])
            s = str(problem_name.split('_')[-1])
            if 'Bimodal' in problem_name:
                problem_index = 7
            else:
                problem_index = 1
                
        elif 'NK' in problem_name:  
            k = str(problem_name.split('_')[-2])
            s = str(problem_name.split('_')[-1])
            problem_index = 2
            instance_name = '../GOMEA_mod/problem_data/n%s_s%s/L%d/%d.txt' % (k, s, dim, r+1)

        elif problem_name == 'MaxCut_Sparse':
            problem_index = 3
            instance_name = '../GOMEA_mod/problem_data/maxcut/set0b/n%.7di%.2d.txt' % (dim, r+1)

        elif problem_name == 'MaxCut_Dense':
            problem_index = 3
            instance_name = '../GOMEA_mod/problem_data/maxcut/set0a/n%.7di%.2d.txt' % (dim, r+1)

        elif problem_name == 'MaxCut_Dense2':
            problem_index = 3
            instance_name = '../GOMEA_mod/problem_data/maxcut/set0e/n%.7di%.2d.txt' % (dim, r+1)

        elif problem_name == 'MaxCut_Sparse2':
            problem_index = 3
            instance_name = '../GOMEA_mod/problem_data/maxcut/set0c/n%.7di%.2d.txt' % (dim, r+1)

        elif problem_name == 'Hierarhical_If-And-Only-If':
            problem_index = 4

        elif problem_name == 'MAXSAT':
            problem_index = 8
            instance_name = 'SAT/uf%d/uf%d-0%d.cnf' % (dim, dim, r+1)

        elif problem_name == 'Spinglass':
            problem_index = 9
            instance_name = 'SPIN/%d/%d_%d' % (dim, dim, r+1)

        problem_index = str(problem_index)

        folder = '../GOMEA_mod/results_population_free_full/%s/%s/' % (settings, problem_name)
        instance_folder = folder + 'L_%d/' % (dim)        
        if r == 0 and os.path.exists(instance_folder):
           shutil.rmtree(instance_folder)

        randomSeed = 42 + r+1
        folder_name = instance_folder + 'run_%d' % r

        callCommand = ['./DSMGA2IMS', str(dim), '2', str(problem_index), '200', str(3600*24*4), str(vtr), str(randomSeed), str(folder_name)]
        if problem_index in ['2','3','8','9']: #NK or MaxCut or MAXSAT or SpinGlass:
            callCommand += [instance_name]
        if problem_index in ['1','2','7']: #NK or MaxCut:
            callCommand += [k, s]
        print(callCommand)
        subprocess.call(callCommand)

        n_evals, elitist_value = get_data(folder_name)
        n_evals_array.append(n_evals)
        elitist_value_array.append(elitist_value)
        #print (elitist_value_array, n_evals_array)

        #if np.unique(elitist_value_array).shape[0] != 1:
        #    print('different values of elitists!')
        #    return False, np.median(n_evals_array), np.median(elitist_value_array)
        print(elitist_value_array, vtr)
        if elitist_value_array[-1] < vtr:
            print ('vtr not reached!')
            return False, np.median(n_evals_array), np.median(elitist_value_array)

    print('n evals array', n_evals_array)
    return True, np.median(n_evals_array), np.median(elitist_value_array)

def run_for_problem(args):
    settings, problem_index, num_runs = args['settings'], args['problem_index'], args['num_runs']
    problem_name = PROBLEM_NAMES[problem_index]
    
    for dim in DIMS[problem_index]:        
        solved = False       
        solved, n_evals, median_elitist = solve_instance(num_runs, settings, problem_name, dim)
        if not solved:
            break
            

from multiprocessing import Pool

def runInParallel(settings, problems):
    args_tuples = []
    for setting in settings:
        for problem_index in problems:
            args_tuples.append({'settings': setting, 'problem_index':problem_index, 'num_runs': args_global.N_RUNS})
                
    print(args_tuples)
    #args_tuples = sorted(args_tuples)
    print('total to run:',len(args_tuples))
    pool = Pool(processes=int(args_global.NUM_PROC))
    pool.map(run_for_problem, args_tuples)



settings = set(['double_edged_DSMGA-II-IMS'])
#print(len(settings))

problems = list(PROBLEM_NAMES.keys())
#problems = [4,6,9,11,14,15,16]
problems = [16]
print(problems, settings)
runInParallel(settings, problems)
