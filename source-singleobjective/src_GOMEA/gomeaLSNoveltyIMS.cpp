#include <iostream>
#include <fstream>
using namespace std;

#include "gomeaLSNoveltyIMS.hpp"
#include "utils.hpp"

gomeaLSNoveltyIMS::gomeaLSNoveltyIMS(Config *config_): GOMEA(config_)
{
  prepareFolder(config->folder);
  initElitistFile(config->folder, config->populationScheme, config->populationSize);
  config->writeOverviewToFile();

  maximumNumberOfGOMEAs         = config->maximumNumberOfGOMEAs;
  IMSsubgenerationFactor        = config->IMSsubgenerationFactor;
  basePopulationSize            = config->basePopulationSize;
  subscheme                     = config->populationScheme;
  currentPopulationSize         = basePopulationSize;
  currentRunCount               = 1;
  numberOfGOMEAs                = 0;
  numberOfGenerationsIMS        = 0;
  minimumGOMEAIndex             = 0;
  numberOfGenerationsIMS        = 0;
  generationsWithoutImprovement = 0;
  step                          = 0;

  createProblemInstance(config->problemIndex, config->numberOfVariables, config, &problemInstance, config->problemInstancePath);
  #ifdef DEBUG
    cout << "Problem Instance created! Problem number is " << config->problemIndex << endl;
  #endif

  sharedInformationInstance = new sharedInformation(config->maxArchiveSize);
  #ifdef DEBUG
    cout << "Shared Information instance created!\n";
  #endif

  //if (config->useVTR)
  //  sharedInformationInstance->vtr = readVTR();

}

gomeaLSNoveltyIMS::~gomeaLSNoveltyIMS()
{
  for (int i = 0; i < numberOfGOMEAs; ++i)
    delete GOMEAs[i];

  delete problemInstance;
  delete sharedInformationInstance;
}


void gomeaLSNoveltyIMS::run()
{
  #ifdef DEBUG
    cout << "IMS started!\n";
  #endif

  bool emittedMaxGomeasHit = false;

  while(!checkTermination())
  {
    if (numberOfGOMEAs < maximumNumberOfGOMEAs || (subscheme == -2 && numberOfGOMEAs * currentRunCount < maximumNumberOfGOMEAs))
    {
      initializeNewGOMEA();
    }
    else if (!emittedMaxGomeasHit)
    {
      cout << "Hit maximum number of simultaneous GOMEAs" << endl;
      emittedMaxGomeasHit = true;
    }

    generationalStepAllGOMEAs();

    numberOfGenerationsIMS++;
  }
}

bool gomeaLSNoveltyIMS::checkTermination()
{
  int i;

  if (numberOfGOMEAs == maximumNumberOfGOMEAs)
  {
    for (i = 0; i < maximumNumberOfGOMEAs; i++)
    {
      if (!GOMEAs[i]->terminated)
        return false;
    }

    return true;
  }
  
  return false;
}

void gomeaLSNoveltyIMS::initializeNewGOMEA()
{
  #ifdef DEBUG
    cout << "Current number Of GOMEAs is " << numberOfGOMEAs << " | Creating New GOMEA!\n";
  #endif


  PopulationNovelty *newPopulation = NULL;
  for (size_t i = 0; i < currentRunCount; ++i)
  {
    newPopulation = new PopulationNovelty(config, problemInstance, sharedInformationInstance, numberOfGOMEAs, currentPopulationSize);
    
    GOMEAs.push_back(newPopulation);
    numberOfGOMEAs++;
  }

  if (subscheme >= -1)
  {
    this->currentPopulationSize *= 2;
  }
  else if (subscheme == -2)
  {
    if (step % 2 == 0)
    {
      this->currentPopulationSize *= 2;
    }
    else
    {
      this->currentRunCount *= 2;
      // this->currentPopulationSize += this->currentPopulationSize / 2;
      this->currentPopulationSize += 16;
    }
  }

  step += 1;
}

void gomeaLSNoveltyIMS::generationalStepAllGOMEAs()
{
  int GOMEAIndexSmallest, GOMEAIndexBiggest;

  GOMEAIndexBiggest  = numberOfGOMEAs - 1;
  GOMEAIndexSmallest = 0;
  while(GOMEAIndexSmallest <= GOMEAIndexBiggest)
  {
    if (!GOMEAs[GOMEAIndexSmallest]->terminated)
      break;

    GOMEAIndexSmallest++;
  }

  if (subscheme == -1)
  {
    generationalStepSmallestGOMEAUntilAllTerminated(GOMEAIndexSmallest, GOMEAIndexBiggest);
  }
  else if (subscheme >= 0)
  {
    GOMEAGenerationalStepAllGOMEAsRecursiveFold(GOMEAIndexSmallest, GOMEAIndexBiggest);
  }
  else if (subscheme == -2)
  {
    generationalStepAllEvenlyGOMEAUntilAllTerminated(GOMEAIndexSmallest, GOMEAIndexBiggest);
  }
}

void gomeaLSNoveltyIMS::generationalStepSmallestGOMEAUntilAllTerminated(int GOMEAIndexSmallest, int GOMEAIndexBiggest)
{
  for (int GOMEAIndex = GOMEAIndexSmallest; GOMEAIndex <= GOMEAIndexBiggest; ++GOMEAIndex)
  {
    cout << "Starting population of size: " << GOMEAs[GOMEAIndex]->populationSize << "." << endl;
    while (!GOMEAs[GOMEAIndex]->terminated)
    {
      GOMEAs[GOMEAIndex]->calculateAverageFitness();
      double fitness_before = sharedInformationInstance->elitist.fitness;

      GOMEAs[GOMEAIndex]->makeOffspring();

      GOMEAs[GOMEAIndex]->copyOffspringToPopulation();

      GOMEAs[GOMEAIndex]->endGeneration();

      GOMEAs[GOMEAIndex]->calculateAverageFitness();
      double fitness_after = sharedInformationInstance->elitist.fitness;

      GOMEAs[GOMEAIndex]->numberOfGenerations++;

      GOMEAs[GOMEAIndex]->terminated = checkTerminationGOMEA(GOMEAIndex);

      if (GOMEAs[GOMEAIndex]->numberOfGenerations >= config->maxGenerations)
      {
        cout << "Max Generations hit!" << endl;
        GOMEAs[GOMEAIndex]->terminated = true;
      }
    }
  }
}

void gomeaLSNoveltyIMS::generationalStepAllEvenlyGOMEAUntilAllTerminated(int GOMEAIndexSmallest, int GOMEAIndexBiggest)
{
  bool any_unterminated = true;
  cout << "Starting population of size " << currentPopulationSize << " and " << currentRunCount << " simultaneous runs." << endl;

  if (GOMEAIndexSmallest > GOMEAIndexBiggest)
  {
    cout << "!!!" << endl;
    exit(1);
  }

  while (any_unterminated)
  {
    any_unterminated = false;
    for (int GOMEAIndex = GOMEAIndexSmallest; GOMEAIndex <= GOMEAIndexBiggest; ++GOMEAIndex)
    {
      any_unterminated |= !GOMEAs[GOMEAIndex]->terminated;

      // Skip over terminated subpopulations.
      if (GOMEAs[GOMEAIndex]->terminated) continue;

      GOMEAs[GOMEAIndex]->calculateAverageFitness();
      double fitness_before = sharedInformationInstance->elitist.fitness;

      GOMEAs[GOMEAIndex]->makeOffspring();

      GOMEAs[GOMEAIndex]->copyOffspringToPopulation();

      GOMEAs[GOMEAIndex]->endGeneration();

      GOMEAs[GOMEAIndex]->calculateAverageFitness();
      double fitness_after = sharedInformationInstance->elitist.fitness;

      GOMEAs[GOMEAIndex]->numberOfGenerations++;

      GOMEAs[GOMEAIndex]->terminated = checkTerminationGOMEA(GOMEAIndex);

      if (GOMEAs[GOMEAIndex]->numberOfGenerations >= config->maxGenerations)
      {
        cout << "Max Generations hit!" << endl;
        GOMEAs[GOMEAIndex]->terminated = true;
      }
    }
  }
}

void gomeaLSNoveltyIMS::GOMEAGenerationalStepAllGOMEAsRecursiveFold(int GOMEAIndexSmallest, int GOMEAIndexBiggest)
{
  int i, GOMEAIndex;

  for(i = 0; i < IMSsubgenerationFactor-1; i++)
  {
    for(GOMEAIndex = GOMEAIndexSmallest; GOMEAIndex <= GOMEAIndexBiggest; GOMEAIndex++)
    {
      if(!GOMEAs[GOMEAIndex]->terminated)
        GOMEAs[GOMEAIndex]->terminated = checkTerminationGOMEA(GOMEAIndex);

      //#if DEBUG || DEBUG_GOM
      //  cout << "GOMEA #" << GOMEAIndex << " terminated=" << GOMEAs[GOMEAIndex]->terminated << endl;
      //#endif

      if((!GOMEAs[GOMEAIndex]->terminated) && (GOMEAIndex >= minimumGOMEAIndex))
      {
        GOMEAs[GOMEAIndex]->calculateAverageFitness();
        double fitness_before = sharedInformationInstance->elitist.fitness;
        //cout << GOMEAIndex << " | avgFitness " << GOMEAs[GOMEAIndex]->averageFitness << endl;

        GOMEAs[GOMEAIndex]->makeOffspring();

        GOMEAs[GOMEAIndex]->copyOffspringToPopulation();

        GOMEAs[GOMEAIndex]->endGeneration();

        GOMEAs[GOMEAIndex]->calculateAverageFitness();
        double fitness_after = sharedInformationInstance->elitist.fitness;
        //cout <<  GOMEAs[GOMEAIndex]->numberOfGenerations << " " << sharedInformationInstance->numberOfEvaluations / 1000000 << endl;
        // if (fitness_after <= fitness_before)
        // {
        //   //cout << "fitness after " << fitness_after << " equals to fitness before"  << fitness_before << " terminating\n";
        //   generationsWithoutImprovement++;
        //   cout << "generations without improvement: " << generationsWithoutImprovement << endl;
        //   if (generationsWithoutImprovement >= config->maxGenerationsWithoutImprovement)
        //     GOMEAs[GOMEAIndex]->terminated = true;
        // }
        // else
        //   generationsWithoutImprovement = 0;
        // cout << GOMEAIndex << " generation " << GOMEAs[GOMEAIndex]->numberOfGenerations << " | avgFitness " << GOMEAs[GOMEAIndex]->averageFitness << " " << sharedInformationInstance->numberOfEvaluations << endl;

        GOMEAs[GOMEAIndex]->numberOfGenerations++;

        if (GOMEAs[GOMEAIndex]->numberOfGenerations >= config->maxGenerations)
          GOMEAs[GOMEAIndex]->terminated = true;
      }
    }

    for(GOMEAIndex = GOMEAIndexSmallest; GOMEAIndex < GOMEAIndexBiggest; GOMEAIndex++)
      GOMEAGenerationalStepAllGOMEAsRecursiveFold(GOMEAIndexSmallest, GOMEAIndex);
  }
}

bool gomeaLSNoveltyIMS::checkTerminationGOMEA(int GOMEAIndex)
{
  for (int i = GOMEAIndex+1; i < numberOfGOMEAs; i++)
  {    
    if (GOMEAs[i]->averageFitness > GOMEAs[GOMEAIndex]->averageFitness &&
      abort_smaller_populations_with_worse_average_fitness)
    {
      // cout << "Terminated GOMEA " << GOMEAIndex << " as average fitness " << GOMEAs[GOMEAIndex]->averageFitness << " is worse than population " << i << " with average fitness: " << GOMEAs[GOMEAIndex]->averageFitness << "." << endl;
      minimumGOMEAIndex = GOMEAIndex+1;
      return true;
    }
  }

  if (config->noImprovementStretchLimit > 0)
  {
    for (size_t i = 1; i < GOMEAs[GOMEAIndex]->populationSize; i++)
    {
      if (GOMEAs[GOMEAIndex]->noImprovementStretches[i] < config->noImprovementStretchLimit)
      {
        return false;
      }
    }
    // Maybe!
    cout << "Improvement Stretch Limit hit!" << endl;
    return true;
  }

  for (size_t i = 1; i < GOMEAs[GOMEAIndex]->populationSize; i++)
  {
    auto individual = GOMEAs[GOMEAIndex]->population[i];
    auto individual_n = GOMEAs[GOMEAIndex]->getPopulationGraphForIndex(i);
    Individual *reference;
    
    if (perform_cluster_aware_convergence_detection)
    {
      if (individual_n.size() == 0)
      {
        // std::cout << "Missing neighborhood..." << std::endl;
        continue;
      }
      else
      {
        reference = GOMEAs[GOMEAIndex]->population[individual_n[0]];
      }
    }
    else
    {
      reference = GOMEAs[GOMEAIndex]->population[0];
    }

    for (size_t j = 0; j < config->numberOfVariables; j++)
    {
      if (individual->genotype[j] != reference->genotype[j])
        return false;
    }
  }

  cout << "CONVERGED!\n";
  return true;
}
