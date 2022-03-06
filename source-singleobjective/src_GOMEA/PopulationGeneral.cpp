#include "PopulationGeneral.hpp"

PopulationGeneral::PopulationGeneral(Config *config_, Problem *problemInstance_, sharedInformation *sharedInformationPointer_, size_t populationIndex_, size_t populationSize_): 
    config(config_), 
    problemInstance(problemInstance_),
    sharedInformationPointer(sharedInformationPointer_),
    populationIndex(populationIndex_), 
    populationSize(populationSize_)
{
    terminated = false;
    numberOfGenerations = 0;
    averageFitness = 0.0;
    
    population.resize(populationSize);
    offspringPopulation.resize(populationSize);
    noImprovementStretches.resize(populationSize);
    
    vector<int> allGenes(config->numberOfVariables);
    iota(allGenes.begin(), allGenes.end(), 0);

    for (size_t i = 0; i < populationSize; ++i)
    {
      noImprovementStretches[i] = 0;

      population[i] = new Individual(config->numberOfVariables, config->alphabetSize);
      population[i]->randomInit(&config->rng);
      evaluateSolution(population[i]);

      if (config->hillClimber == 1)
        hillClimberSingle(population[i]);
      else if (config->hillClimber == 2)
        hillClimberMultiple(population[i]);
      else if (config->hillClimber == 3)
        hillClimberMultiple(population[i]);

      offspringPopulation[i] = new Individual(config->numberOfVariables, config->alphabetSize);
      *offspringPopulation[i] = *population[i];      
    }

    createFOSInstance(config->FOSIndex, &FOSInstance, config->numberOfVariables, config->alphabetSize, config->similarityMeasure, problemInstance);
}

void PopulationGeneral::calculateAverageFitness()
{
	averageFitness = -INFINITY;
	for (size_t i = 0; i < populationSize; ++i)
		averageFitness += population[i]->fitness;
	averageFitness /= populationSize;
}

void PopulationGeneral::copyOffspringToPopulation()
{
  for(size_t i = 0; i < populationSize; i++)
  {
  	*population[i] = *offspringPopulation[i];
  }
}

void PopulationGeneral::tournamentSelection(int k, vector<Individual*> &population, vector<Individual*> &offspringPopulation)
{
  int populationSize = population.size();

  vector<int> indices(populationSize * k);
  for (int i = 0; i < k; ++i)
  {
    for (int j = 0; j < populationSize; ++j)
      indices[populationSize*i + j] = j;

    shuffle(indices.begin() + populationSize*i, indices.begin() + populationSize*(i+1), config->rng);
    // for (int i = 0; i < indices.size(); ++i)
    //   cout <<indices[i] << " ";
    // cout << endl;

  }
  for (int i = 0; i < populationSize; i++)
  {
    int winnerInd = 0;
    double winnerFitness = -1e+308;

    for (int j = 0; j < k; j++)
    {
      int challengerInd = indices[k*i+j];
      double challengerFitness = population[challengerInd]->fitness;
      //cout << i << " " << j << " " << challengerInd << endl;
      if (challengerFitness > winnerFitness)
      {
        winnerInd = challengerInd;
        winnerFitness = challengerFitness;
      }
    }

    *offspringPopulation[i] = *population[winnerInd];
  }
  // for (int i = 0; i < populationSize; i++)
  //   cout << i << " " << *population[i] << endl;

  //cout << endl;
}

void PopulationGeneral::hillClimberSingle(Individual *solution)
{
	vector<int> positions(config->numberOfVariables);
  iota(positions.begin(), positions.end(), 0);   

  shuffle(positions.begin(), positions.end(), config->rng);

  for (int j = 0; j < positions.size(); ++j)
  {
    int curPos = positions[j];
    char curValue = solution->genotype[curPos];

  	for (char k = 0; k < config->alphabetSize[curPos]; ++k)
  	{
    	if (k == curValue)
      		continue;

    	Individual backup = *solution;  
    	vector<int> touchedGenes(1, curPos);

    	solution->genotype[curPos] = k;

    	evaluateSolution(solution);

    	if (solution->fitness <= backup.fitness)
      		*solution = backup;
    }
  }
}

void PopulationGeneral::hillClimberMultiple(Individual *solution)
{
	vector<int> positions(config->numberOfVariables);
	iota(positions.begin(), positions.end(), 0);

	while (true)
	{
	  bool solutionImproved = false;

	  shuffle(positions.begin(), positions.end(), config->rng);

	  for (int j = 0; j < positions.size(); ++j)
	  {
	    int curPos = positions[j];
	    char curValue = solution->genotype[curPos];

	    for (char k = 0; k < config->alphabetSize[curPos]; ++k)
	    {
	      if (k == curValue)
	        continue;

	      Individual backup = *solution;  
	      vector<int> touchedGenes(1, curPos);

	      solution->genotype[curPos] = k;

	      evaluateSolution(solution);

	      if (solution->fitness > backup.fitness)
	        solutionImproved = true;
	      else
	        *solution = backup;
	    }
	  }

	  if (!solutionImproved)
	    break;
	}
}

void PopulationGeneral::findNeighbors(vector<vector<int> > &neighbors)
{
  vector<vector<int> > &graphPointer = FOSInstance->graph;
  
  neighbors.resize(FOSInstance->FOSSize()); 
  
  vector<vector<int> > neighborsMatrix(FOSInstance->FOSSize());

  for (size_t i = 0; i < FOSInstance->FOSSize(); i++)
  {
    neighbors[i].clear();
    neighborsMatrix[i].resize(config->numberOfVariables);
    fill(neighborsMatrix[i].begin(), neighborsMatrix[i].end(), 0);

    if (FOSInstance->FOSElementSize(i) == 0 || FOSInstance->FOSElementSize(i) == config->numberOfVariables)
      continue;

    for(size_t j = 0; j < FOSInstance->FOSElementSize(i); j++)
    {
      int variableFromFOS = FOSInstance->FOSStructure[i][j]; 
      neighborsMatrix[i][variableFromFOS] = -1;
    }

    for(size_t j = 0; j < FOSInstance->FOSElementSize(i); j++)
    {
      int variableFromFOS = FOSInstance->FOSStructure[i][j]; 
    
      for (size_t k = 0; k < graphPointer[variableFromFOS].size(); ++k)
      {
        int neighbor = graphPointer[variableFromFOS][k];
        if (neighborsMatrix[i][neighbor] >= 0)
          neighborsMatrix[i][neighbor] += 1;
        //cout << "neighbor " << neighbor << endl;
      }
    }

    for(size_t j = 0; j < config->numberOfVariables; j++)
    {
      //cout << (double)neighborsMatrix[i][j]/(double)FOSInstance->FOSElementSize(i) << " ";
      //if (neighborsMatrix[i][j] <= 0)
      //  continue;
      
      if (neighborsMatrix[i][j] >  0)
        neighbors[i].push_back(j);  
      
      //if ((double)neighborsMatrix[i][j]/(double)FOSInstance->FOSElementSize(i) > config->thresholdStrength)
      //  neighbors[i].push_back(j);  
        //cout << "neighbor " << neighbor << endl;
    }
    // cout << endl;
    // cout << i << " " << neighbors[i].size() << ": ";
    // for (int j = 0; j < neighbors[i].size(); ++j)
    //   cout << neighbors[i][j] << " ";
    // cout << endl;
  }
}


bool PopulationGeneral::FI(size_t offspringIndex, Individual *backup)
{
  vector<int> FOSIndices;
  FOSInstance->orderFOS(config->orderFOS, FOSIndices, &config->rng); 

  bool solutionHasChanged = 0;

  for (size_t i = 0; i < FOSInstance->FOSSize(); i++)
  {
    int ind = FOSIndices[i];

    if (FOSInstance->FOSElementSize(ind) == 0 || FOSInstance->FOSElementSize(ind) == config->numberOfVariables)
      continue;

    vector<int> touchedGenes;      
    bool donorEqualToOffspring = true;
    for(size_t j = 0; j < FOSInstance->FOSElementSize(ind); j++)
    {
      int variableFromFOS = FOSInstance->FOSStructure[ind][j];
      offspringPopulation[offspringIndex]->genotype[variableFromFOS] = sharedInformationPointer->elitist.genotype[variableFromFOS];
      touchedGenes.push_back(variableFromFOS);
      if (backup->genotype[variableFromFOS] != offspringPopulation[offspringIndex]->genotype[variableFromFOS])
        donorEqualToOffspring = false;
    }

    if (!donorEqualToOffspring)
    {
      evaluateSolution(offspringPopulation[offspringIndex]);

      if (offspringPopulation[offspringIndex]->fitness > backup->fitness)
      {
        *backup = *offspringPopulation[offspringIndex];
        solutionHasChanged = true;
      }
      else
      {
        *offspringPopulation[offspringIndex] = *backup;
      }
    }
    if (solutionHasChanged)
      break;
  }

  if (!solutionHasChanged)
  {
    *offspringPopulation[offspringIndex] = sharedInformationPointer->elitist;
  }

  return solutionHasChanged;
}

bool PopulationGeneral::conditionalFI(size_t offspringIndex, Individual *backup, vector<vector<int> > &neighbors)
{
  vector<bool> sampled(config->numberOfVariables, false);
  bool solutionHasChanged = 0;

  vector <bool> dependentMonitor(config->numberOfVariables, 0);

  vector<int> FOSIndices;
  FOSInstance->orderFOS(config->orderFOS, FOSIndices, &config->rng); 


  for (size_t i = 0; i < FOSInstance->FOSSize(); i++)
  {
    int ind = FOSIndices[i];

    if (FOSInstance->FOSElementSize(ind) == 0 || FOSInstance->FOSElementSize(ind) == config->numberOfVariables)
      continue;

    fill(dependentMonitor.begin(), dependentMonitor.end(), 0);
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

    for(size_t j = 0; j < FOSInstance->FOSElementSize(ind); j++)
    {
      int variableFromFOS = FOSInstance->FOSStructure[ind][j];    
      sampled[variableFromFOS]=1;  
    }
    
    // for(size_t j = 0; j < neighbors[ind].size(); j++)
    // {
    //   int neighbor = neighbors[ind][j];
    //   if (sampled[neighbor])
    //   {
    //     if (find(dependent.begin(), dependent.end(), neighbor) == dependent.end())
    //       dependent.push_back(neighbor);  
    //   }
    // }
        
    
    bool canBeChosen = true;
    for (int k = 0; k < dependent.size(); ++k)
    {
      if (offspringPopulation[offspringIndex]->genotype[dependent[k]] != sharedInformationPointer->elitist.genotype[dependent[k]])
      {
        canBeChosen = false;
        break;
      }
    }
    if (!canBeChosen)
      continue;

    vector<int> touchedGenes;     
    bool donorEqualToOffspring = true; 
    for(size_t j = 0; j < FOSInstance->FOSElementSize(ind); j++)
    {
      int variableFromFOS = FOSInstance->FOSStructure[ind][j];
      offspringPopulation[offspringIndex]->genotype[variableFromFOS] = sharedInformationPointer->elitist.genotype[variableFromFOS];
      touchedGenes.push_back(variableFromFOS);
      //sampled[variableFromFOS] = 1;
      if (backup->genotype[variableFromFOS] != offspringPopulation[offspringIndex]->genotype[variableFromFOS])
        donorEqualToOffspring = false;
    }

    if (!donorEqualToOffspring)
    {
      evaluateSolution(offspringPopulation[offspringIndex]);

      if (offspringPopulation[offspringIndex]->fitness > backup->fitness)
      {
        *backup = *offspringPopulation[offspringIndex];
        solutionHasChanged = true;
      }
      else
      {
        *offspringPopulation[offspringIndex] = *backup;
      }
    }
    if (solutionHasChanged)
      break;
  }

  if (!solutionHasChanged)
  {
    *offspringPopulation[offspringIndex] = sharedInformationPointer->elitist;
  }

  return solutionHasChanged;
}


void PopulationGeneral::evaluateSolution(Individual *solution)
{
  checkTimeLimit();

  archiveRecord searchResult;
  
  if (config->saveEvaluations)
    sharedInformationPointer->evaluatedSolutions->checkAlreadyEvaluated(solution->genotype, &searchResult);
  
  if (searchResult.isFound)
  {
    solution->fitness = searchResult.value;
  }
  else
  {
    problemInstance->calculateFitness(solution);
    if (config->problemIndex == 10)
      sharedInformationPointer->numberOfEvaluations = problemInstance->getEvals();
    else
      sharedInformationPointer->numberOfEvaluations++;

    if (config->saveEvaluations)
      sharedInformationPointer->evaluatedSolutions->insertSolution(solution->genotype, solution->fitness);

    updateElitistAndCheckVTR(solution);

    if (sharedInformationPointer->numberOfEvaluations >= config->maxEvaluations)
    {
      cout << "Max evals limit reached! Terminating...\n";
      throw customException("max evals");
    }
  }
}

void PopulationGeneral::checkTimeLimit()
{
  if (getMilliSecondsRunningSinceTimeStamp(sharedInformationPointer->startTimeMilliseconds) > config->timelimitSeconds*1000)
  {
    cout << "TIME LIMIT REACHED!" << endl;
    throw customException("time");
  }
}

void PopulationGeneral::updateElitistAndCheckVTR(Individual *solution)
{
  if (sharedInformationPointer->firstEvaluationEver || (solution->fitness > sharedInformationPointer->elitist.fitness))
  {
    sharedInformationPointer->elitistSolutionHittingTimeMilliseconds = getMilliSecondsRunningSinceTimeStamp(sharedInformationPointer->startTimeMilliseconds);
    sharedInformationPointer->elitistSolutionHittingTimeEvaluations = sharedInformationPointer->numberOfEvaluations;

    sharedInformationPointer->elitist = *solution;

    /* Check the VTR */
    if (solution->fitness >= config->vtr)
    {
      writeElitistSolutionToFile(config->folder, sharedInformationPointer->elitistSolutionHittingTimeEvaluations, sharedInformationPointer->elitistSolutionHittingTimeMilliseconds, solution);
      cout << "VTR HIT!\n";
      throw customException("vtr");
    }
  
    writeElitistSolutionToFile(config->folder, sharedInformationPointer->elitistSolutionHittingTimeEvaluations, sharedInformationPointer->elitistSolutionHittingTimeMilliseconds, solution);
  }


  sharedInformationPointer->firstEvaluationEver = false;
}

int PopulationGeneral::compareSolutions(Individual *x, Individual*y)
{
  if (x->fitness > y->fitness)
    return 1;
  if (x->fitness == y->fitness)
    return 0;
  return -1;
}
