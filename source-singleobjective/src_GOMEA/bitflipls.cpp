//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
//
// This research code was modified as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
//
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt (modifications for relevant article), Arkadiy Dushatskiy (original developer)

#include <algorithm>

using namespace std;

#include "bitflipls.hpp"

#include "Population.hpp"
#include "utils.hpp"

BitflipLS::BitflipLS(Config *config_, sharedInformation *info_, Problem *problem_): config(config_), info(info_), problem(problem_)
{
    // For BitflipLS to work, the problem needs to be binary.
    assert(all_of(config->alphabetSize.begin(), 
        config->alphabetSize.end(), 
        [](int s) { return s == 2; }));
    // Initialize the sequence.
    sequence.resize(config->numberOfVariables);
    iota(sequence.begin(), sequence.end(), 0);
    
}

BitflipLS::~BitflipLS()
{
    // WARNING: Both problemInstance as well as
    // sharedInformationInstance are assumed to be shared
    // by another approach.
    // (eg. we should not delete them here.)
    // Furthermore, Config is cleaned up elsewere.
}

// void BitflipLS::shuffle_order(mt19937 *rng)
// {
//     shuffle(sequence.front(), sequence.back(), rng);
// }

void BitflipLS::optimize(Individual *individual_, size_t populationSize)
{
    // BitflipLS always changes one item at a time.
    vector<int> changedGenes = {0};

    Individual *current_individual = individual_;
    evaluateSolution(current_individual, NULL, changedGenes, populationSize);

    // Keep a backup for partial evaluations.
    Individual *backup_individual = new Individual(individual_->genotype, 
        individual_->fitness, 
        individual_->behavior);

    bool improved = true;
    while (improved) 
    {
        improved = false;
        for (auto it = sequence.begin(); it != sequence.end(); ++it) {
            char original_gene = current_individual->genotype[*it];
            double original_fitness = current_individual->fitness;

            // char new_gene = config->alphabet[original_gene == config->alphabet[0]];
            char new_gene = !original_gene;

            current_individual->genotype[*it] = new_gene;

            changedGenes[0] = *it;
            evaluateSolution(current_individual, backup_individual, changedGenes, populationSize);
            double new_fitness = current_individual->fitness;

            if (new_fitness < original_fitness)
            {
                // Rollback non-improving move.
                current_individual->genotype[*it] = original_gene;
                current_individual->fitness = original_fitness;
                #if DEBUG
                cout << "Failed to improve @ "<< *it << ". New fitness (" << new_fitness << ") < original fitness (" << original_fitness << ").\n";
                #endif
            } else {
                // Update original solution.
                backup_individual->genotype[*it] = new_gene;
                backup_individual->fitness = new_fitness;
                #if DEBUG
                cout << "Improved @ " << *it << ". New fitness (" << new_fitness << ") > original fitness (" << original_fitness << ").\n";
                #endif
                improved |= new_fitness > original_fitness;
            }
        }
    }
    // Clean up the supplementary solution.
    delete backup_individual;
}


/// Copied and modified from PopulationGeneral.cpp


void BitflipLS::evaluateSolution(Individual *new_solution, Individual *orig_solution, vector<int> &changedGenes, size_t populationSize)
{
  checkTimeLimit();

  archiveRecord searchResult;
  
  if (config->saveEvaluations)
    info->evaluatedSolutions->checkAlreadyEvaluated(new_solution->genotype, &searchResult);
  
  if (searchResult.isFound)
  {
    new_solution->fitness = searchResult.value;
  }
  else
  {
    if (orig_solution == NULL)
    {
        problem->calculateFitness(new_solution);
    } 
    else 
    {
        problem->calculateFitnessPartialEvaluations(new_solution, orig_solution, changedGenes, orig_solution->fitness);
    }

    if (config->problemIndex == 10)
      info->numberOfEvaluations = problem->getEvals();
    else
      info->numberOfEvaluations++;

    if (config->saveEvaluations)
      info->evaluatedSolutions->insertSolution(new_solution->genotype, new_solution->fitness);

    checkVTR(new_solution, populationSize);

    if (info->numberOfEvaluations >= config->maxEvaluations)
    {
      cout << "Max evals limit reached! Terminating...\n";
      throw customException("max evals");
    }
  }
}

void BitflipLS::checkTimeLimit()
{
  if (getMilliSecondsRunningSinceTimeStamp(info->startTimeMilliseconds) > config->timelimitSeconds*1000)
  {
    cout << "TIME LIMIT REACHED!" << endl;
    throw customException("time");
  }
}

void BitflipLS::checkVTR(Individual *solution, size_t populationSize)
{
  bool improved = info->firstEvaluationEver || (solution->fitness > info->elitist.fitness);
  
  bool vtr_hit = (solution->fitness >= config->vtr);
  bool key_sol = false;

  bool use_vtr = config->vtr_n_unique > 0;
  bool use_key = config->vtr_n_unique < 0;
  // Only check if it is a key solution if that is actually used!
  if (use_key)
    key_sol = problem->isKeySolution(solution->genotype, solution->fitness);
  bool unique_vtr_hit = false;
  bool unique_key_sol_hit = false;
  if (improved || vtr_hit || key_sol)
  {
    info->elitistSolutionHittingTimeMilliseconds = getMilliSecondsRunningSinceTimeStamp(info->startTimeMilliseconds);
    info->elitistSolutionHittingTimeEvaluations = info->numberOfEvaluations;

    if (improved)
      info->elitist = *solution;

    // cout << "Improved: " << improved << "; Use VTR: " << use_vtr << "; VTR hit: " << vtr_hit << "; Use Key: " << use_key << "; Key Solution:" << key_sol << endl;

    /* Check the VTR */
    /* If VTR count is negative, use key_sol instead. */
    if ((use_vtr && vtr_hit) || (use_key && key_sol))
    {
      archiveRecord rec;
      info->vtrUniqueSolutions->checkAlreadyEvaluated(solution->genotype, &rec);
      if (!rec.isFound)
      {
        info->vtrUniqueSolutions->insertSolution(solution->genotype, solution->fitness);
        
        // Stop once all unique occurences have been hit.
        if (info->vtrUniqueSolutions->archive.size() >= abs(config->vtr_n_unique))
        {
          cout << "LAST UNIQUE VTR HIT!" << endl;
          writeElitistSolutionToFile(config->folder, info->elitistSolutionHittingTimeEvaluations, info->elitistSolutionHittingTimeMilliseconds, solution, true, populationSize);
          throw customException("vtr");
        } else {
          unique_vtr_hit = true;
          cout << "UNIQUE VTR HIT!" << endl;
        }
      }
      
    }

    if (improved || unique_vtr_hit )
      writeElitistSolutionToFile(config->folder, info->elitistSolutionHittingTimeEvaluations, info->elitistSolutionHittingTimeMilliseconds, solution, unique_vtr_hit, populationSize);
  }


  info->firstEvaluationEver = false;
}