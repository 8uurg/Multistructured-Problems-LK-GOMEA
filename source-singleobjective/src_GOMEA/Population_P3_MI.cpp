//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
//
// This research code was modified as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
//
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt (modifications for relevant article), Arkadiy Dushatskiy (original developer)

#include "Population_P3_MI.hpp"


Population_P3_MI::~Population_P3_MI()
{
  for (size_t i = 0; i < populationSize; ++i)
  {
    delete population[i];
    delete offspringPopulation[i];
  }

  delete FOSInstance;
}

void Population_P3_MI::makeOffspring()
{
  if (config->hillClimber == 3)
    return;
  
  int someSolutionsImproved = false;

  for (int i = 0; i < populationSize; ++i)
  {
    if (noImprovementStretches[i] == 0) // solution improved or zero level
    {
      sharedInformationPointer->pyramid->insertSolution(currentPyramidLevel, population[i]->genotype, population[i]->fitness);
      someSolutionsImproved = true;
    }
  }

  if (currentPyramidLevel == sharedInformationPointer->pyramid->levels.size() || someSolutionsImproved == false)
  {
    terminated = true;
    return;
  }

  vector<Individual*> populationForModelLearning(sharedInformationPointer->pyramid->levels[currentPyramidLevel].size());
  for (int i = 0; i < populationForModelLearning.size(); ++i)
  {
    populationForModelLearning[i] = new Individual(config->numberOfVariables, config->alphabetSize);
    *populationForModelLearning[i] = *sharedInformationPointer->pyramid->levels[currentPyramidLevel][i];
  }

  if (config->tournamentSelection)
    tournamentSelection(2, sharedInformationPointer->pyramid->levels[currentPyramidLevel], populationForModelLearning); //performs tournament selection and saves the winners to population array

  if (config->FOSIndex == 1 || config->FOSIndex == 3 || config->FOSIndex == 4)
    FOSInstance->learnFOS(populationForModelLearning, NULL, &config->rng);
  else if (config->FOSIndex == 2) //Fixed LT based on problem VIG
    FOSInstance->learnFOS(populationForModelLearning, &problemInstance->graph, &config->rng);

  for (int i = 0; i < populationForModelLearning.size(); ++i)
    delete populationForModelLearning[i];

  FOSInstance->setCountersToZero();
  if (config->AnalyzeFOS)
  {
    FOSInstance->writeToFileFOS(config->folder, populationIndex, currentPyramidLevel);
    if (config->FOSIndex == 1)
      FOSInstance->writeMIMatrixToFile(config->folder, populationIndex, currentPyramidLevel);
  }

  if (config->conditionalGOM == 1)
    FOSInstance->buildGraph(config->MI_threshold, &config->rng);
  
  generateOffspring();

  if (config->AnalyzeFOS)
    FOSInstance->writeFOSStatistics(config->folder, populationIndex, currentPyramidLevel);
}

void Population_P3_MI::generateOffspring()
{
  vector<vector<int> > neighbors;
  if (config->conditionalGOM > 0)
    findNeighbors(neighbors);

  for(size_t i = 0; i < populationSize; i++)
  {
      Individual backup = *population[i];  
      
      bool solutionHasChanged;
      if (config->conditionalGOM == 0)
        solutionHasChanged = GOM(i, &backup);
      else
        solutionHasChanged = conditionalGOM(i, &backup, neighbors);

      /* Phase 2 (Forced Improvement): optimal mixing with elitist solution */
      if (config->useForcedImprovements)
      {
        if ((!solutionHasChanged) || (noImprovementStretches[i] > (1+(log(populationSize)/log(10)))))
        {
          if (config->conditionalGOM == 0)
            FI(i, &backup);
          else
            conditionalFI(i, &backup, neighbors);
                      
           #ifdef DEBUG_GOM
             cout << "after FI " << i << " " << *offspringPopulation[i] << endl;
           #endif    
        }    
      }

    //if(!(offspringPopulation[i]->fitness > population[i]->fitness))
    
    if (compareSolutions(offspringPopulation[i], population[i]) != 1)
      noImprovementStretches[i]++;
    else
      noImprovementStretches[i] = 0;
  }
}

bool Population_P3_MI::GOM(size_t offspringIndex, Individual *backup)
{
  size_t donorIndex;
  bool solutionHasChanged = false;
  bool thisIsTheElitistSolution = *offspringPopulation[offspringIndex] == sharedInformationPointer->elitist;//(sharedInformationPointer->elitistSolutionpopulationIndex == populationIndex) && (sharedInformationPointer->elitistSolutionOffspringIndex == offspringIndex);
  
  *offspringPopulation[offspringIndex] = *population[offspringIndex];

  vector<int> FOSIndices;
  FOSInstance->orderFOS(config->orderFOS, FOSIndices, &config->rng); 

  vector<int> donorIndices(sharedInformationPointer->pyramid->levels[currentPyramidLevel].size());
  iota(donorIndices.begin(), donorIndices.end(), 0);

  for (size_t i = 0; i < FOSInstance->FOSSize(); i++)
  {
    int ind = FOSIndices[i];

    if (FOSInstance->FOSElementSize(ind) == 0 || FOSInstance->FOSElementSize(ind) == config->numberOfVariables)
      continue;

    bool donorEqualToOffspring = true;
    int indicesTried = 0;

    while (donorEqualToOffspring && indicesTried < donorIndices.size())
    {
      int j = config->rng() % (donorIndices.size() - indicesTried);
      //cout << "rand ind:" << j << " " << indicesTried+j << " " << donorIndices.size() << endl;
      swap(donorIndices[indicesTried], donorIndices[indicesTried + j]);
      donorIndex = donorIndices[indicesTried];
      indicesTried++;
      
      if (offspringPopulation[offspringIndex]->genotype == sharedInformationPointer->pyramid->levels[currentPyramidLevel][donorIndex]->genotype)
        continue;

      vector<int> touchedGenes;
      for(size_t j = 0; j < FOSInstance->FOSElementSize(ind); j++)
      {
        int variableFromFOS = FOSInstance->FOSStructure[ind][j];      
        offspringPopulation[offspringIndex]->genotype[variableFromFOS] = sharedInformationPointer->pyramid->levels[currentPyramidLevel][donorIndex]->genotype[variableFromFOS];
        touchedGenes.push_back(variableFromFOS);

        if (backup->genotype[variableFromFOS] != offspringPopulation[offspringIndex]->genotype[variableFromFOS])
          donorEqualToOffspring = false;      
      }

      //cout << "donorEqualToOffspring: " << donorEqualToOffspring << endl;

      if (!donorEqualToOffspring)
      {
        evaluateSolution(offspringPopulation[offspringIndex]);

        // accept the change if this solution is not the elitist and the fitness is at least equally good (allows random walk in neutral fitness landscape)
        // however, if this is the elitist solution, only accept strict improvements, to avoid convergence problems
        
        //if ((!thisIsTheElitistSolution && (offspringPopulation[offspringIndex]->fitness >= backup->fitness)) || 
        //    (thisIsTheElitistSolution && (offspringPopulation[offspringIndex]->fitness > backup->fitness)))   
        if ((!thisIsTheElitistSolution && compareSolutions(offspringPopulation[offspringIndex], backup) >= 0) || 
          (thisIsTheElitistSolution && compareSolutions(offspringPopulation[offspringIndex], backup) == 1))         
        {       
          *backup = *offspringPopulation[offspringIndex];
          solutionHasChanged = true;
          FOSInstance->improvementCounters[ind]++;
        }
        else
        {
          *offspringPopulation[offspringIndex] = *backup;
        }

        FOSInstance->usageCounters[ind]++;
      }

      if (!config->donorSearch) //if not exhaustive donor search then stop searching anyway
        break;
    }
  }
  return solutionHasChanged;
}


bool Population_P3_MI::conditionalGOM(size_t offspringIndex, Individual *backup, vector<vector<int> > &neighbors)
{
  size_t donorIndex;
  bool solutionHasChanged = false;
  bool thisIsTheElitistSolution = *offspringPopulation[offspringIndex] == sharedInformationPointer->elitist;//(sharedInformationPointer->elitistSolutionpopulationIndex == populationIndex) && (sharedInformationPointer->elitistSolutionOffspringIndex == offspringIndex);
  
  *offspringPopulation[offspringIndex] = *population[offspringIndex];

  //#ifdef DEBUG_GOM
  //cout << "before gom " << offspringIndex << " " << *offspringPopulation[offspringIndex] << " " << thisIsTheElitistSolution << endl;  
  //#endif 

  vector<bool> sampled(config->numberOfVariables, false);
  vector<int> donorIndices(sharedInformationPointer->pyramid->levels[currentPyramidLevel].size());
  iota(donorIndices.begin(), donorIndices.end(), 0);
  vector<int> FOSIndices;
  FOSInstance->orderFOS(config->orderFOS, FOSIndices, &config->rng); 

  vector <bool> dependentMonitor(config->numberOfVariables, 0);

  for (size_t i = 0; i < FOSInstance->FOSSize(); i++)
  {
    int ind = FOSIndices[i];
    fill(dependentMonitor.begin(), dependentMonitor.end(), 0);

    if (FOSInstance->FOSElementSize(ind) == 0 || FOSInstance->FOSElementSize(ind) == config->numberOfVariables)
      continue;

    vector<int> dependent;
    for(size_t j = 0; j < neighbors[ind].size(); j++)
    {
      int neighbor = neighbors[ind][j];
      if (sampled[neighbor])
      {
        if (dependentMonitor[neighbor] == 0)
          dependent.push_back(neighbor); 
        dependentMonitor[neighbor]=1; 
      }
    }
    //cout << dependent.size() << endl;
    //if (offspringPopulation[offspringIndex]->genotype[dependent[k]] != sharedInformationPointer->pyramid->levels[currentPyramidLevel][curDonorIndex].first[dependent[k]])
       
    int indicesTried = 0;
    bool donorFound = false;
    
    for(size_t j = 0; j < FOSInstance->FOSElementSize(ind); j++)
    {
      int variableFromFOS = FOSInstance->FOSStructure[ind][j];    
      sampled[variableFromFOS]=1;  
    }

    while (indicesTried < donorIndices.size())
    {
      int j = config->rng() % (donorIndices.size() - indicesTried);
      //cout << "rand ind:" << j << " " << indicesTried+j << " " << donorIndices.size() << endl;
      swap(donorIndices[indicesTried], donorIndices[indicesTried + j]);
      donorIndex = donorIndices[indicesTried];
      indicesTried++;

      if (donorIndex == offspringIndex)
        continue;

      bool dependentAreEqual = true;
      for (int k = 0; k < dependent.size(); ++k)
      {
        if (offspringPopulation[offspringIndex]->genotype[dependent[k]] != sharedInformationPointer->pyramid->levels[currentPyramidLevel][donorIndex]->genotype[dependent[k]])
        {
          dependentAreEqual = false;
          break;
        }
      }
      
      if (dependentAreEqual)
      {
        //cout << "dependentAreEqual" << endl;
        if (!config->donorSearch) // if not exhaustive search, stop
        {
          donorFound = true;
          break;
        }
        //if exhaustive donor search, check if donor is equal to offspring
        bool donorEqualToOffspring = true;
        for(size_t k = 0; k < FOSInstance->FOSElementSize(ind); k++)
        {
          int variableFromFOS = FOSInstance->FOSStructure[ind][k];      
          if (sharedInformationPointer->pyramid->levels[currentPyramidLevel][donorIndex]->genotype[variableFromFOS] != offspringPopulation[offspringIndex]->genotype[variableFromFOS])
          {
            donorEqualToOffspring = false;    
            break;
          }
        }
        if (!donorEqualToOffspring)
        {
          //cout << "donorFound" << endl;
          donorFound = true;
          break;
        }
      }

    }
    
    if (!donorFound)
      continue;

    bool donorEqualToOffspring = true;
    vector<int> touchedGenes;
    for(size_t j = 0; j < FOSInstance->FOSElementSize(ind); j++)
    {
      int variableFromFOS = FOSInstance->FOSStructure[ind][j];     
      offspringPopulation[offspringIndex]->genotype[variableFromFOS] = sharedInformationPointer->pyramid->levels[currentPyramidLevel][donorIndex]->genotype[variableFromFOS];
      touchedGenes.push_back(variableFromFOS);
      if (backup->genotype[variableFromFOS] != offspringPopulation[offspringIndex]->genotype[variableFromFOS])
        donorEqualToOffspring = false;    
      // sampled[variableFromFOS]=1;  
    }

    if (!donorEqualToOffspring)
    {
      //for(size_t j = 0; j < FOSInstance->FOSElementSize(ind); j++)
      //  sampled[FOSInstance->FOSStructure[ind][j]] = 1;

      evaluateSolution(offspringPopulation[offspringIndex]);

      // accept the change if this solution is not the elitist and the fitness is at least equally good (allows random walk in neutral fitness landscape)
      // however, if this is the elitist solution, only accept strict improvements, to avoid convergence problems
      //if ((!thisIsTheElitistSolution && (offspringPopulation[offspringIndex]->fitness >= backup->fitness)) || 
      //    (thisIsTheElitistSolution && (offspringPopulation[offspringIndex]->fitness > backup->fitness)))   
      if ((!thisIsTheElitistSolution && compareSolutions(offspringPopulation[offspringIndex], backup) >= 0) || 
          (thisIsTheElitistSolution && compareSolutions(offspringPopulation[offspringIndex], backup) == 1))   
      {       
        *backup = *offspringPopulation[offspringIndex];
        
        solutionHasChanged = true;

        FOSInstance->improvementCounters[ind]++;
      }
      else
      {
        *offspringPopulation[offspringIndex] = *backup;
      }

      FOSInstance->usageCounters[ind]++;
    }
  }

  // for (int i = 0; i < config->numberOfVariables; ++i)
  //   cout << sampled[i];
  // cout << endl;  

  return solutionHasChanged;
}

