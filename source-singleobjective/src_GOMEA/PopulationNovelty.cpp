#include "PopulationNovelty.hpp"


PopulationNovelty::~PopulationNovelty()
{
  for (size_t i = 0; i < populationSize; ++i)
  {
    delete population[i];
    delete offspringPopulation[i];
  }

  // Clean up the novelty archive.
  // size_t s = noveltyArchive.size();
  // for (size_t i = 0; i < s; ++s)
  // {
  //   delete noveltyArchive[i];
  // }

  delete FOSInstance;
  for (size_t fos_i = 0; fos_i < FOSs.size(); ++fos_i)
  {
    delete FOSs[fos_i];
  }
}

void PopulationNovelty::learnFOS(FOS *FOSInstance)
{
  vector<Individual*> populationForModelLearning(populationSize);
  for (int i = 0; i < populationSize; ++i)
  {
    populationForModelLearning[i] = new Individual(config->numberOfVariables, config->alphabetSize);
    *populationForModelLearning[i] = *population[i];
  }

  if (config->tournamentSelection)
    tournamentSelection(2, population, populationForModelLearning); //performs tournament selection and saves the winners to population array

  if (config->FOSIndex == 0 || config->FOSIndex == 1  || config->FOSIndex == 3  || config->FOSIndex == 4  || config->FOSIndex == 5)
    FOSInstance->learnFOS(populationForModelLearning, NULL, &config->rng);

  for (int i = 0; i < populationSize; ++i)
    delete populationForModelLearning[i];

  FOSInstance->setCountersToZero();
  if (config->AnalyzeFOS)
  {
    FOSInstance->writeToFileFOS(config->folder, populationIndex, numberOfGenerations);
    // if (config->FOSIndex == 1)
    FOSInstance->writeMIMatrixToFile(config->folder, populationIndex, numberOfGenerations);
  }

  if (config->conditionalGOM == 1)
    FOSInstance->buildGraph(config->MI_threshold, &config->rng);

  FOSInstance->shrinkMemoryUsage();
}

void PopulationNovelty::learnIndividualFOS(FOS *FOSInstance, size_t population_idx)
{
  vector<int> pg_current_ind = getPopulationGraphForIndex(population_idx);
  vector<Individual*> populationForModelLearning(1 + pg_current_ind.size());
  // populationForModelLearning.reserve(populationSize);

  populationForModelLearning[0] = new Individual(config->numberOfVariables, config->alphabetSize);
  *populationForModelLearning[0] = *population[population_idx];

  for (int gi = 0; gi < pg_current_ind.size(); ++gi)
  {
    int i = pg_current_ind[gi];
    populationForModelLearning[gi + 1] = new Individual(config->numberOfVariables, config->alphabetSize);
    *populationForModelLearning[gi + 1] = *population[i];
  }

  if (config->tournamentSelection)
    tournamentSelection(2, population, populationForModelLearning); //performs tournament selection and saves the winners to population array

  if (config->FOSIndex == 0 || config->FOSIndex == 1  || config->FOSIndex == 3  || config->FOSIndex == 4  || config->FOSIndex == 5)
    FOSInstance->learnFOS(populationForModelLearning, NULL, &config->rng);

  for (int i = 0; i < populationForModelLearning.size(); ++i)
    delete populationForModelLearning[i];

  FOSInstance->setCountersToZero();
  if (config->AnalyzeFOS)
  {
    size_t fos_id = populationIndex * populationSize + population_idx + 1;
    FOSInstance->writeToFileFOS(config->folder, fos_id, numberOfGenerations);
    // if (config->FOSIndex == 1)
    FOSInstance->writeMIMatrixToFile(config->folder, fos_id, numberOfGenerations);
  }

  if (config->conditionalGOM == 1)
    FOSInstance->buildGraph(config->MI_threshold, &config->rng);

  FOSInstance->shrinkMemoryUsage();
}

void PopulationNovelty::learnClusterFOS(FOS *FOSInstance, size_t population_idx)
{
  vector<int> pg_current_ind = getPopulationClusterGraphForIndex(population_idx);
  vector<Individual*> populationForModelLearning(1 + pg_current_ind.size());
  // populationForModelLearning.reserve(populationSize);

  populationForModelLearning[0] = new Individual(config->numberOfVariables, config->alphabetSize);
  *populationForModelLearning[0] = *population[population_idx];

  for (int gi = 0; gi < pg_current_ind.size(); ++gi)
  {
    int i = pg_current_ind[gi];
    populationForModelLearning[gi + 1] = new Individual(config->numberOfVariables, config->alphabetSize);
    *populationForModelLearning[gi + 1] = *population[i];
  }

  if (config->tournamentSelection)
    tournamentSelection(2, population, populationForModelLearning); //performs tournament selection and saves the winners to population array

  if (config->FOSIndex == 0 || config->FOSIndex == 1  || config->FOSIndex == 3  || config->FOSIndex == 4  || config->FOSIndex == 5)
    FOSInstance->learnFOS(populationForModelLearning, NULL, &config->rng);

  for (int i = 0; i < populationForModelLearning.size(); ++i)
    delete populationForModelLearning[i];

  FOSInstance->setCountersToZero();
  if (config->AnalyzeFOS)
  {
    size_t fos_id = populationIndex * populationSize + population_idx + 1;
    FOSInstance->writeToFileFOS(config->folder, fos_id, numberOfGenerations);
    // if (config->FOSIndex == 1)
    FOSInstance->writeMIMatrixToFile(config->folder, fos_id, numberOfGenerations);
    // cout << "Wrote fos " << fos_id << " for Generation " << numberOfGenerations << " containing " << populationForModelLearning.size() << " samples." << endl; 
  }

  if (config->conditionalGOM == 1)
    FOSInstance->buildGraph(config->MI_threshold, &config->rng);

  FOSInstance->shrinkMemoryUsage();
}

void PopulationNovelty::learnIndividualBootstrapFOS(FOS *FOSInstance, size_t population_idx)
{
  // Tournament size
  const size_t k = 3;
  // Number of solutions in the bootstrap.
  const size_t bootstrapSize = populationSize;

  const size_t distance_kind = config->distance_kind;
  
  vector<Individual*> populationForModelLearning(bootstrapSize);
  // populationForModelLearning.reserve(populationSize);
  
  Individual *leader = population[population_idx];

  // Always place leader in the bootstrap population.
  populationForModelLearning[0] = new Individual(config->numberOfVariables, config->alphabetSize);
  *populationForModelLearning[0] = *leader;

  uniform_int_distribution rngpop((size_t) 0, populationSize - 1);

  for (int gi = 1; gi < bootstrapSize; ++gi)
  {
    size_t i = rngpop(config->rng);
    double di = computeDistance(distance_kind, leader, population[i]);
    for (size_t s = 0; s < k; ++s)
    {
      size_t j = rngpop(config->rng);
      double dj = computeDistance(distance_kind, leader, population[j]);
      if (dj < di)
      {
        i = j;
        di = dj;
      }
    }
    
    populationForModelLearning[gi] = new Individual(config->numberOfVariables, config->alphabetSize);
    *populationForModelLearning[gi] = *population[i];
  }

  if (config->tournamentSelection)
    tournamentSelection(2, population, populationForModelLearning); //performs tournament selection and saves the winners to population array

  if (config->FOSIndex == 0 || config->FOSIndex == 1  || config->FOSIndex == 3  || config->FOSIndex == 4  || config->FOSIndex == 5)
    FOSInstance->learnFOS(populationForModelLearning, NULL, &config->rng);

  for (int i = 0; i < populationForModelLearning.size(); ++i)
    delete populationForModelLearning[i];

  FOSInstance->setCountersToZero();
  if (config->AnalyzeFOS)
  {
    size_t fos_id = populationIndex * populationSize + population_idx;
    FOSInstance->writeToFileFOS(config->folder, fos_id, numberOfGenerations);
    // if (config->FOSIndex == 1)
    FOSInstance->writeMIMatrixToFile(config->folder, fos_id, numberOfGenerations);
  }

  if (config->conditionalGOM == 1)
    FOSInstance->buildGraph(config->MI_threshold, &config->rng);

  FOSInstance->shrinkMemoryUsage();
}

void PopulationNovelty::makeOffspring()
{
  if (config->verbosity >= 1)
  {
    cout << "Start generation for population size " << populationSize << endl;
  }

  // Update clusters as well!
  updatePopulationTopology();

  learnFOS(FOSInstance);

  if (config->multiFOS == 1)
  {
    // FOS per individual
    for (size_t i = 0; i < populationSize; ++i)
    {
      learnIndividualFOS(FOSs[i], i);
    }
  }
  else if (config->multiFOS == 2)
  {
    // FOS per cluster representative
    for (size_t i: clusterIndices)
    {
      learnClusterFOS(FOSs[i], i);
    }
  }
  else if (config->multiFOS == 3)
  {
    // FOS per cluster representative, bootstrapped
    for (size_t i: clusterIndices)
    {
      learnIndividualBootstrapFOS(FOSs[i], i);
    }
  }
  
  
  generateOffspring();

  bool echoPopulation = config->verbosity > 2;
  
  if (echoPopulation)
  {
  cout << "Population:\n";
  }
  for (auto it = population.begin(); it != population.end(); ++it)
  {
    Individual *i = *it;
    
    // Print out current population.
    if (echoPopulation)
    {
      cout << " (" << i->fitness << ")\t\t[";
      for (auto g = i->genotype.begin(); g != i->genotype.end(); ++g)
      {
        cout << ((int) *g);
      }
      cout << "]\n";
      cout << "  => (" << i->inherentFitness << ")\t[";
      for (auto g = i->behavior.begin(); g != i->behavior.end(); ++g)
      {
        cout << ((int) *g);
      }
      cout << "]" << endl;
    }
  }

  if (config->AnalyzeFOS)
    FOSInstance->writeFOSStatistics(config->folder, populationIndex, numberOfGenerations);
}


void PopulationNovelty::generateOffspring()
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

    if(!(offspringPopulation[i]->fitness > population[i]->fitness))
      noImprovementStretches[i]++;
    else
      noImprovementStretches[i] = 0;

    //cout << i << " #evals:" << sharedInformationPointer->numberOfEvaluations << " " << improved_restricted << endl;
  }
}

bool PopulationNovelty::GOM(size_t offspringIndex, Individual *backup)
{
  FOS *FOSInstance = this->FOSInstance;
  if (config->multiFOS >= 1)
  {
    FOSInstance = FOSs[populationFOSIndices[offspringIndex]];
  }

  size_t donorIndex;
  bool solutionHasChanged = false;
  Individual &elitist = *getElitist();
  bool thisIsTheElitistSolution = *offspringPopulation[offspringIndex] == elitist;//(sharedInformationPointer->elitistSolutionpopulationIndex == populationIndex) && (sharedInformationPointer->elitistSolutionOffspringIndex == offspringIndex);

  // Distance function for crowding (probably should be moved elsewere?)
  // Current implementation is hamming distance.
  auto distance = [](Individual* a, Individual* b)
  {
    double result = 0;
    size_t n = min(a->numberOfVariables, b->numberOfVariables);
    for (size_t i = 0; i < n; ++i)
    {
      result += a->genotype[i] != b->genotype[i];
    }
    result /= (double) n;
    return result; 
  };

  *offspringPopulation[offspringIndex] = *population[offspringIndex];

  //#ifdef DEBUG_GOM
   // cout << "before gom " << offspringIndex << " " << *offspringPopulation[offspringIndex] << " " << thisIsTheElitistSolution << endl;  
  //#endif 

  vector<int> FOSIndices;
  FOSInstance->orderFOS(config->orderFOS, FOSIndices, &config->rng); 

  // Always full matrix.
  // vector<int> donorIndices(populationSize);
  // iota(donorIndices.begin(), donorIndices.end(), 0);
  
  // Use graph instead
  vector<int> donorIndices = getPopulationGraphForIndex(offspringIndex);  
  // if (true)
  // {
  //   cout << "PopulationGraph Indices: [";
  //   for (auto ind: populationGraph[offspringIndex])
  //     cout << ind << ",";
  //   cout << "]" << endl;
  //   cout << "Donor Indices: [";
  //   for (auto ind: donorIndices)
  //     cout << ind << ",";
  //   cout << "]" << endl;
  // }

  uniform_real_distribution unif(0.0, 1.0);

  for (size_t i = 0; i < FOSInstance->FOSSize(); i++)
  {
    int ind = FOSIndices[i];

    if (FOSInstance->FOSElementSize(ind) == 0 || FOSInstance->FOSElementSize(ind) == config->numberOfVariables)
      continue;

    bool donorEqualToOffspring = true;
    int indicesTried = 0;

    double randomNum = 0.0;

    // Note: only generate a random number if neccesary.
    if (config->mixingOperator == 2)
    {
      randomNum = unif(config->rng);
    }

    while (donorEqualToOffspring && indicesTried < donorIndices.size())
    {
      int j = config->rng() % (donorIndices.size() - indicesTried);
      //cout << "rand ind:" << j << " " << indicesTried+j << " " << donorIndices.size() << endl;
      swap(donorIndices[indicesTried], donorIndices[indicesTried + j]);
      donorIndex = donorIndices[indicesTried];
      indicesTried++;

      if (donorIndex == offspringIndex)
        continue;

      vector<int> touchedGenes;
      for(size_t j = 0; j < FOSInstance->FOSElementSize(ind); j++)
      {
        int variableFromFOS = FOSInstance->FOSStructure[ind][j]; 
        
        // Standard copy-over
        if (config->mixingOperator == 0 || 
          (config->mixingOperator == 2 && randomNum > config->flipProbability))
        {
          offspringPopulation[offspringIndex]->genotype[variableFromFOS] = population[donorIndex]->genotype[variableFromFOS];
          
          if (backup->genotype[variableFromFOS] != offspringPopulation[offspringIndex]->genotype[variableFromFOS])
            donorEqualToOffspring = false;      
        }
        else if (config->mixingOperator == 1 || 
          (config->mixingOperator == 2 && randomNum <= config->flipProbability))
        {
          // Perform block bitflip instead. (does not use a donor)
          offspringPopulation[offspringIndex]->genotype[variableFromFOS] = !offspringPopulation[offspringIndex]->genotype[variableFromFOS];
          donorEqualToOffspring = false;
        }
        touchedGenes.push_back(variableFromFOS);
      }
      Individual *competitor = population[donorIndex];
      if (donorIndex < offspringIndex)
        competitor = offspringPopulation[donorIndex];
      
      double distance_to_backup = 0.0;
      double distance_to_competitor = 0.0;
      
      if (!donorEqualToOffspring && config->crowding > 0)
      {
        distance_to_backup = distance(offspringPopulation[offspringIndex], backup);
        distance_to_competitor = distance(offspringPopulation[offspringIndex], competitor);
      }

      bool closer_to_competitor = distance_to_competitor < distance_to_backup;
      bool conditional_crowding_reject = config->crowding==1 && closer_to_competitor;

      if (!donorEqualToOffspring && !conditional_crowding_reject)
      {
        evaluateSolution(offspringPopulation[offspringIndex]);

        // accept the change if this solution is not the elitist and the fitness is at least equally good (allows random walk in neutral fitness landscape)
        // however, if this is the elitist solution, only accept strict improvements, to avoid convergence problems
        if (((!thisIsTheElitistSolution && (offspringPopulation[offspringIndex]->fitness >= backup->fitness)) || 
            (thisIsTheElitistSolution && (offspringPopulation[offspringIndex]->fitness > backup->fitness))))  
        {       
          *backup = *offspringPopulation[offspringIndex];
          solutionHasChanged = true;
          FOSInstance->improvementCounters[ind]++;
          // cout << "Improvement!" << endl;
        }
        else if (config->crowding == 2 && closer_to_competitor &&
          ((!thisIsTheElitistSolution && (offspringPopulation[offspringIndex]->fitness >= competitor->fitness)) || 
            (thisIsTheElitistSolution && (offspringPopulation[offspringIndex]->fitness > competitor->fitness))))
        {
          // cout << "Distance to backup: " << distance_to_backup << "; Distance to Competitor: " << distance_to_competitor << ". cmp: " << closer_to_competitor << "." << endl;
          // Crowding is turned on, and we have ended up closer to the competitor.
          // Deterministic Crowding dictates that we should replace the closest solution.
          *competitor = *offspringPopulation[offspringIndex];
          // Revert for current solution. This also means the solution hasn't changed...
          *offspringPopulation[offspringIndex] = *backup;
          FOSInstance->improvementCounters[ind]++;
        }
        else
        {
          // cout << "No improvement. Sparsity current: " << backup->fitness << ", new: " << offspringPopulation[offspringIndex]->fitness << "." << endl;

          *offspringPopulation[offspringIndex] = *backup;
        }

        FOSInstance->usageCounters[ind]++;
      }

      if (!config->donorSearch) //if not exhaustive donor search then stop searching anyway
        break;
    }
  }

  //if (thisIsTheElitistSolution)
  //  cout << "after:" << *offspringPopulation[offspringIndex] << endl;
  return solutionHasChanged;
}

bool PopulationNovelty::conditionalGOM(size_t offspringIndex, Individual *backup, vector<vector<int> > &neighbors)
{
  FOS *FOSInstance = this->FOSInstance;
  if (config->multiFOS >= 1)
  {
    FOSInstance = FOSs[populationFOSIndices[offspringIndex]];
  }
  
  size_t donorIndex;
  bool solutionHasChanged = false;
  Individual &elitist = *getElitist();
  bool thisIsTheElitistSolution = *offspringPopulation[offspringIndex] == elitist;//(sharedInformationPointer->elitistSolutionpopulationIndex == populationIndex) && (sharedInformationPointer->elitistSolutionOffspringIndex == offspringIndex);
  
  *offspringPopulation[offspringIndex] = *population[offspringIndex];

  #ifdef DEBUG_GOM
  cout << "before gom " << offspringIndex << " " << *offspringPopulation[offspringIndex] << " " << thisIsTheElitistSolution << endl;  
  #endif 

  vector<bool> sampled(config->numberOfVariables, false);
  vector<int> donorIndices(populationSize);
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
    // cout << "dependent:";
    // for (int j = 0; j < dependent.size(); ++j)
    // cout << dependent[j] << " ";
    // cout << endl;
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
        if (offspringPopulation[offspringIndex]->genotype[dependent[k]] != population[donorIndex]->genotype[dependent[k]])
        {
          dependentAreEqual = false;
          break;
        }
      }
      
      if (dependentAreEqual)
      {
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

          if (offspringPopulation[offspringIndex]->genotype[variableFromFOS] != population[donorIndex]->genotype[variableFromFOS])
          {
            donorEqualToOffspring = false;    
            break;
          }
        }
        if (!donorEqualToOffspring)
        {
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
      offspringPopulation[offspringIndex]->genotype[variableFromFOS] = population[donorIndex]->genotype[variableFromFOS];

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
      if ((!thisIsTheElitistSolution && (offspringPopulation[offspringIndex]->fitness >= backup->fitness)) || 
          (thisIsTheElitistSolution && (offspringPopulation[offspringIndex]->fitness > backup->fitness)))   
      {       
        *backup = *offspringPopulation[offspringIndex];
        
        solutionHasChanged = true;

        //for(size_t j = 0; j < FOSInstance->FOSElementSize(ind); j++)
        //  sampled[FOSInstance->FOSStructure[ind][j]]=1;
        FOSInstance->improvementCounters[ind]++;
      }
      else
      {
        *offspringPopulation[offspringIndex] = *backup;
      }

      FOSInstance->usageCounters[ind]++;
    }

    // bool filled = true;
    // for (int i = 0; i < config->numberOfVariables; ++i)
    // {
    //   if (sampled[i] == 0)
    //   {
    //     filled = false;
    //     break;
    //   }
    // }
    // if (filled)
    //   fill(sampled.begin(), sampled.end(),0);
  }

  return solutionHasChanged;
}


