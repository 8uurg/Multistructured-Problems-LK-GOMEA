/***************************************************************************
 *   Copyright (C) 2015 Tian-Li Yu and Shih-Huan Hsu                       *
 *   tianliyu@ntu.edu.tw                                                   *
 ***************************************************************************/

#pragma once

#include <math.h>
#include <iostream>
#include <cstdlib>
#include <string>
#include <time.h>
#include <vector>
#include "statistics.h"
#include "dsmga2.h"
#include "global.h"
#include "chromosome.h"
#include "problems.h"
#include "utils.h"
using namespace std;

int main_verbosity = 0;

void generationalStepAllPopulations(vector<DSMGA2*> &populations, vector<bool> &terminated, vector<int> &numberOfGenerations);
void GenerationalStepAllRecursiveFold(int IndexSmallest, int IndexBiggest, vector<DSMGA2*> &populations, vector<bool> &terminated, vector<int> &numberOfGenerations);
bool checkTerminationGOMEA(int Index, vector<DSMGA2*> &populations, vector<bool> &terminated);

void generationalStepAllPopulations(vector<DSMGA2*> &populations, vector<bool> &terminated, vector<int> &numberOfGenerations)
{
  int IndexSmallest, IndexBiggest;

  IndexBiggest  = populations.size() - 1;
  IndexSmallest = 0;
  while(IndexSmallest <= IndexBiggest)
  {
    if (!terminated[IndexSmallest])
      break;

    IndexSmallest++;
  }

  GenerationalStepAllRecursiveFold(IndexSmallest, IndexBiggest, populations, terminated, numberOfGenerations);
}

void GenerationalStepAllRecursiveFold(int IndexSmallest, int IndexBiggest, vector<DSMGA2*> &populations, vector<bool> &terminated, vector<int> &numberOfGenerations)
{
  int i, Index, IMSsubgenerationFactor = 4;

  for(i = 0; i < IMSsubgenerationFactor-1; i++)
  {
    for(Index = IndexSmallest; Index <= IndexBiggest; Index++)
    {
      if(!terminated[Index] && Index >= Chromosome::minimumGOMEAIndex)
        terminated[Index] = checkTerminationGOMEA(Index, populations, terminated);
    //terminated[Index] = populations[Index]->shouldTerminate(Index, populations, terminated);

      if(!terminated[Index] && Index >= Chromosome::minimumGOMEAIndex)
      {
        if (main_verbosity >= 1)
          cout << "population #" << Index << " | num gens " << numberOfGenerations[Index] << " | terminated=" << terminated[Index] << endl;
            
        populations[Index]->generation = numberOfGenerations[Index];
        populations[Index]->oneRun(false);

        if (Chromosome::elitistFitness >= populations[Index]->population[0].getMaxFitness())
        {
            if (main_verbosity >= 1)
              cout << "VTR HIT!\n";
            throw customException("vtr");
        }

        numberOfGenerations[Index]++;

        if (numberOfGenerations[Index] >= 200)
          terminated[Index] = true;
      }
    }

    for(Index = IndexSmallest; Index < IndexBiggest; Index++)
      GenerationalStepAllRecursiveFold(IndexSmallest, Index, populations, terminated, numberOfGenerations);
  }
}

bool checkTerminationGOMEA(int Index, vector<DSMGA2*> &populations, vector<bool> &terminated)
{
  for (int i = Index+1; i < populations.size(); i++)
  {    
    if (main_verbosity >= 1)
      cout << "mean fitness #" << i << " " << populations[i]->stFitness.getMean() << endl;
    if (populations[i]->stFitness.getMean() > populations[Index]->stFitness.getMean())
    {
      Chromosome::minimumGOMEAIndex = Index+1;
      if (main_verbosity >= 1)
        cout << "smaller fitness:" << i << ":" << populations[i]->stFitness.getMean() << " | " << Index << ":" << populations[Index]->stFitness.getMean() << endl;
      return true;
    }
  }

  return populations[Index]->shouldTerminate();
}

bool print_usage_description = false;

int
main (int argc, char *argv[]) {
    if (print_usage_description)
    {
          printf ("DSMGA2 ell nInitial function maxGen max_evaluations time_limit vtr rand_seed folder [instancePath] [k] [s]\n");
          printf ("function: \n");
          printf ("     Trap:  1\n");
          printf ("     ADF    :  2\n");
          printf ("     MaxCut :  3\n");
          printf ("     HIFF   :  4\n");
          printf ("     Bimodal Trap    :  7\n");
          printf ("     MAXSAT    :  8\n");
          printf ("     Spinglass    :  9\n");
          printf ("     Best-of-Traps    : 10\n");
    }

    int maxFe = atoi(argv[5]); // max fe
    Chromosome::maxnfe = maxFe;
    int display = 1;
    
    int ell = atoi (argv[1]); // problem size
    int nInitial = atoi (argv[2]); // initial population size
    int fffff = atoi (argv[3]); // function
    int maxGen = atoi (argv[4]); // max generation
    Chromosome::generationsWithoutImprovement = 0;

    long long time_limit_seconds = atoi(argv[6]); // time limit in seconds
    Chromosome::timeLimitMilliseconds = time_limit_seconds*1000;
    Chromosome::startTimestamp = getCurrentTimeStampInMilliSeconds();

    double vtr = atof(argv[7]); //value to reach
    Chromosome::vtr = vtr;

    int rand_seed = atoi (argv[8]);  // rand seed
    
    //cout << argv[9] << endl;
    string folder = string(argv[9]);  // folder
    if (fffff == 2 || fffff == 3 || fffff == 8  || fffff == 9 || fffff == 10 || fffff == 11) //ADF or MaxCut or MAXSAT os Spinglass or Best-of-Traps
    {
        // Note: Best-of-Traps is instance based as it uses a random number generator, which can differ between compilers.

        string instancePath = string(argv[10]);  // Path to instance
        Chromosome::instancePath = instancePath;
    }
    if (fffff == 1 || fffff == 2 || fffff == 7) //Conccatenated Trap or ADF or bimodal concatenated trap
    {
        int k, s;
        if (fffff == 2)
        {
            k = atoi(argv[11]); // subfunction size
            s = atoi(argv[12]);  // step    
        }
        else
        {
            k = atoi(argv[10]); // subfunction size
            s = atoi(argv[11]);  // step    
        }

        Chromosome::k = k;
        Chromosome::s = s;
    }

    if (main_verbosity >= 1)
      cout << folder << " vtr:" << vtr << endl;
    prepareFolder(folder);
    initElitistFile(folder, nInitial);
    Chromosome::folder = folder;

    // if (fffff == 4) {

    //     char filename[200];
    //     //sprintf(filename, "./NK_Instance/pnk%d_%d_%d_%d", ell, 4, 1, 1);
    //     sprintf(filename, "./NK_Instance/pnk%d_%d_%d_%d", ell, 4, 5 , 1);

    //     if (SHOW_BISECTION) printf("Loading: %s\n", filename);
    //     FILE *fp = fopen(filename, "r");
    //     loadNKWAProblem(fp, &nkwa);
    //     fclose(fp);
    // }

    // if (fffff == 5) {
    //     char filename[200];
    //     sprintf(filename, "./SPIN/%d/%d_%d",ell, ell, 1);
    //     if (SHOW_BISECTION) printf("Loading: %s\n", filename);
    //     loadSPIN(filename, &mySpinGlassParams);
    // }

    // if (fffff == 6) {
    //     char filename[200];
    //     sprintf(filename, "./SAT/uf%d/uf%d-0%d.cnf", ell, ell, 1);
    //     if (SHOW_BISECTION) printf("Loading: %s\n", filename);
    //     loadSAT(filename, &mySAT);
    // }

    ////////////////////////////////////////////////////////////////////////////////////
    createProblemInstance(fffff, ell, &(Chromosome::problemInstance), Chromosome::instancePath, Chromosome::k, Chromosome::s);

    ////////////////////////////////////////////////////////////////////////////////////
    
    if (rand_seed != -1)  // time
        myRand.seed((unsigned long)rand_seed);

    int i;

    Statistics stGen, stFE, stLSFE;
    int usedGen;

    int failNum = 0;
    vector<DSMGA2*> populations;
    vector<bool> terminated;
    vector<int> numberOfGenerations;
    int pop_size = nInitial;
    
    Chromosome::function = (Chromosome::Function)fffff;
    Chromosome::nfe = 0;
    Chromosome::lsnfe = 0;
    Chromosome::hitnfe = 0;
    Chromosome::hit = false;
    Chromosome::elitist = new unsigned long [ell];
    Chromosome::elitistFitness = -1e+308;
    Chromosome::minimumGOMEAIndex = 0;

    try
    {
        while (populations.size() < 100)
        {
            if (main_verbosity >= 1)
              cout << ell << endl;
            DSMGA2 *ga = new DSMGA2(ell, pop_size, maxGen, maxFe, fffff);            
            populations.push_back(ga);
            terminated.push_back(false);
            numberOfGenerations.push_back(0);

            generationalStepAllPopulations(populations, terminated, numberOfGenerations);

            pop_size *= 2;
            if (main_verbosity >= 1)
              cout << populations.size() << " " << pop_size << endl;
        }
    }
    catch (customException &ex)
    {}
    if (main_verbosity >= 1)
      cout<<endl; 
    //printf ("%f  %f  %f %d\n", stGen.getMean (), stFE.getMean(), stLSFE.getMean(), failNum);

    //if (fffff == 4) freeNKWAProblem(&nkwa);
    for (int i = 0; i < populations.size(); ++i)
        delete populations[i];

    delete[] Chromosome::elitist;

    return EXIT_SUCCESS;
}
