//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
//
// This research code was modified as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
//
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt (modifications for relevant article), Arkadiy Dushatskiy (original developer)

#include "PopulationNovelGeneral.hpp"
#include "Linkup.hpp"
#include <queue>
#include <random>

/* BEGIN: CHANGES FOR MEMORY USAGE REDUCTION */
void PopulationNovelGeneral::upsizeDistanceMatrix()
{
  for (size_t i = 0; i < populationSize; ++i)
  {
    populationDistances[i].resize(populationSize);
    fill(populationDistances[i].begin(), populationDistances[i].end(), 0.0);
  }
}

void PopulationNovelGeneral::downsizeDistanceMatrix()
{
  for (size_t i = 0; i < populationSize; ++i)
  {
    populationDistances[i].resize(0);
    // Actually free the memory.
    populationDistances[i].shrink_to_fit();
  }
}

void PopulationNovelGeneral::downsizePopulationGraph()
{
  for (size_t i = 0; i < populationGraph.size(); ++i)
  {
    populationGraph[i].resize(0);
    // Actually free the memory.
    populationGraph[i].shrink_to_fit();
  }
}

void PopulationNovelGeneral::downsizePopulationClusterGraph()
{
  for (size_t i = 0; i < populationClusterGraph.size(); ++i)
  {
    populationClusterGraph[i].resize(0);
    // Actually free the memory.
    populationClusterGraph[i].shrink_to_fit();
  }
}

void PopulationNovelGeneral::endGeneration()
{
  downsizeDistanceMatrix();
  downsizePopulationGraph();
  downsizePopulationClusterGraph();
}

vector<int> PopulationNovelGeneral::getPopulationGraphForIndex(int population_idx)
{
  if (population_graph_is_full_graph)
  {
    vector<int> result(populationSize);
    iota(result.begin(), result.end(), 0);
    return result;
  }
  else
  {
    return this->populationGraph[population_idx];
  }
}

vector<int> PopulationNovelGeneral::getPopulationClusterGraphForIndex(int population_idx)
{
  if (population_cluster_graph_is_full_graph)
  {
    vector<int> result(populationSize);
    iota(result.begin(), result.end(), 0);
    return result;
  }
  else
  {
    return this->populationClusterGraph[population_idx];
  }
}
/* END: CHANGES FOR MEMORY USAGE REDUCTION */

PopulationNovelGeneral::PopulationNovelGeneral(Config *config_, Problem *problemInstance_, sharedInformation *sharedInformationPointer_, size_t populationIndex_, size_t populationSize_): 
    config(config_), 
    problemInstance(problemInstance_),
    sharedInformationPointer(sharedInformationPointer_),
    populationIndex(populationIndex_), 
    populationSize(populationSize_)
{
    terminated = false;
    numberOfGenerations = 0;
    averageFitness = 0.0;

    // Ensure initial elitist is arbitrarily bad such that the first evaluation part of this population will replace it.
    firstEvaluationPopulation = true;
    populationElitist.fitness = -INFINITY;
    
    population.resize(populationSize);
    offspringPopulation.resize(populationSize);
    noImprovementStretches.resize(populationSize);

    populationGraph.resize(populationSize);
    populationDistances.resize(populationSize);
    population_graph_is_full_graph = false;
    if (config->populationTopology == 0 || config->populationTopology == 18) 
    {
      population_graph_is_full_graph = true;
    }
    // for (size_t i = 0; i < populationSize; ++i)
    // {
      // Don't reserve the full graph, takes too much space!
      // populationGraph[i].resize(populationSize);
      // iota(populationGraph[i].begin(), populationGraph[i].end(), 0);
    // }
    // Full distance graph initialization has been moved!
    clusterGraphEqualsPopulationGraph();
    
    vector<int> allGenes(config->numberOfVariables);
    iota(allGenes.begin(), allGenes.end(), 0);

    bool useProbabilisticCompleteInitialization = true;

    // This loop is split such that sparsity can be computed
    // Over the entire population.
    // evaluateSolution with the second argument will not immediately compute
    // sparsity.
    if (useProbabilisticCompleteInitialization)
    {
      // Initialize population randomly, but ensuring every variable is equally
      // distributed (has the same number of zeroes and ones, for example)
      vector<size_t> populationPermutation(populationSize);
      iota(populationPermutation.begin(), populationPermutation.end(), 0);
      // Construct population
      for (size_t i = 0; i < populationSize; ++i)
      {
        population[i] = new Individual(config->numberOfVariables, config->alphabetSize);
      }
      for (size_t v = 0; v < config->numberOfVariables; ++v)
      {
        // Shuffle!
        shuffle(populationPermutation.begin(), populationPermutation.end(), config->rng);
        for (size_t pi = 0; pi < populationSize; ++pi)
        {
          size_t i = populationPermutation[pi];
          population[i]->genotype[v] = pi % config->alphabetSize[v];
        }
      }
      if (config->p_sample_reference > 0.0) {
        if (! config->reference_solution.has_value()) {
          cerr << "When requesting sampling from a reference solution, place provide a reference solution." << endl;
          exit(1);
        }
        vector<char>& reference_solution = config->reference_solution.value();
        uniform_real_distribution unif01(0.0, 1.0);
        
        // Replace a fraction of the genes with a the value from the reference solution
        // (ignoring the specialized initialization)
        for (size_t i = 0; i < populationSize; ++i)
        {
          for (size_t v = 0; v < config->numberOfVariables; ++v)
          {
            if (unif01(config->rng) < config-> p_sample_reference)
              population[i]->genotype[v] = reference_solution[v];
          }
        }
      }
      for(size_t i = 0; i < populationSize; ++i)
      {
        evaluateSolution(population[i]);
      }
    }
    else
    {
      for (size_t i = 0; i < populationSize; ++i)
      {
        noImprovementStretches[i] = 0;

        population[i] = new Individual(config->numberOfVariables, config->alphabetSize);
        population[i]->randomInit(&config->rng);

        evaluateSolution(population[i]);
      }
    }
    for (size_t i = 0; i < populationSize; ++i)
    {
      if (config->hillClimber == 1)
        hillClimberSingle(population[i]);
      else if (config->hillClimber == 2)
        hillClimberMultiple(population[i]);
      else if (config->hillClimber == 3)
        hillClimberMultiple(population[i]);

      offspringPopulation[i] = new Individual(config->numberOfVariables, config->alphabetSize);
      *offspringPopulation[i] = *population[i];
    }
    for (size_t i = 0; i < populationSize; ++i)
    {
      *offspringPopulation[i] = *population[i];
    }

    // updatePopulationTopology();

    // Global FOS...
    // The conditional variant uses it, would take some work to modify.
    createFOSInstance(config->FOSIndex, &FOSInstance, config->numberOfVariables, config->alphabetSize, config->similarityMeasure, problemInstance);

    populationFOSIndices.resize(populationSize);

    if (config->multiFOS == 0)
    {
      // Default: 1 FOS.
      FOSs.resize(1);
      createFOSInstance(config->FOSIndex, &FOSs[0], config->numberOfVariables, config->alphabetSize, config->similarityMeasure, problemInstance);
      
      fill(populationFOSIndices.begin(), populationFOSIndices.end(), 0);

    }
    else if (config->multiFOS == 1)
    {
      // FOS for each Individual.
      FOSs.resize(populationSize);
      for (size_t fos_i = 0; fos_i < FOSs.size(); ++fos_i)
      {
        createFOSInstance(config->FOSIndex, &FOSs[fos_i], config->numberOfVariables, config->alphabetSize, config->similarityMeasure, problemInstance);
      }
      iota(populationFOSIndices.begin(), populationFOSIndices.end(), 0);
    }
    else if (config->multiFOS == 2 || config->multiFOS == 3)
    {
      // FOS per cluster... Initialize per Individual
      // (cluster selects Individual as representative)
      // Note: 2 simply uses the cluster as population,
      // while 3 creates a bootstrap population based on distance.
      // (Dirk's idea!)
      // TODO: Make this maybe more efficient in terms of memory.
      FOSs.resize(populationSize);
      for (size_t fos_i = 0; fos_i < FOSs.size(); ++fos_i)
      {
        createFOSInstance(config->FOSIndex, &FOSs[fos_i], config->numberOfVariables, config->alphabetSize, config->similarityMeasure, problemInstance);
      }
      // Every solution has their own FOS.
      fill(populationFOSIndices.begin(), populationFOSIndices.end(), 0);      
    }

}

Individual* PopulationNovelGeneral::getElitist()
{
    if (config->use_elitist_per_population == 0)
    {
      // Original approach
      return &sharedInformationPointer->elitist;
    }
    else
    {
      return &populationElitist;
    }
}
void PopulationNovelGeneral::setElitist(Individual &other)
{
  *getElitist() = other;
  if (config->use_elitist_per_population == 1)
  {
    firstEvaluationPopulation = false;
  }
}
bool PopulationNovelGeneral::hasElitist()
{
  if (config->use_elitist_per_population == 0)
  {
    // Original approach
    return !sharedInformationPointer->firstEvaluationEver;
  }
  else
  {
    return !firstEvaluationPopulation;
  }
}


void PopulationNovelGeneral::calculateAverageFitness()
{
	averageFitness = 0.0;
	for (size_t i = 0; i < populationSize; ++i)
		averageFitness += population[i]->fitness;
	averageFitness /= populationSize;
}

void PopulationNovelGeneral::copyOffspringToPopulation()
{
  for(size_t i = 0; i < populationSize; i++)
  {
  	*population[i] = *offspringPopulation[i];
  }
}

void PopulationNovelGeneral::tournamentSelection(int k, vector<Individual*> &population, vector<Individual*> &offspringPopulation)
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

void PopulationNovelGeneral::hillClimberSingle(Individual *solution)
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

void PopulationNovelGeneral::hillClimberMultiple(Individual *solution)
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

void PopulationNovelGeneral::findNeighbors(vector<vector<int> > &neighbors)
{
  // TODO: Yet to implement: Make this aware of the population graph.

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


bool PopulationNovelGeneral::FI(size_t offspringIndex, Individual *backup)
{
  // TODO: Modify FI to completely operate with my modifications...

  vector<int> FOSIndices;

  size_t FOSidx = populationFOSIndices[offspringIndex];
  // Locally override the instance FOSInstance.
  FOS *FOSInstance = FOSs[FOSidx];
  FOSInstance->orderFOS(config->orderFOS, FOSIndices, &config->rng); 

  Individual &elitist = *getElitist();
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
      offspringPopulation[offspringIndex]->genotype[variableFromFOS] = elitist.genotype[variableFromFOS];
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
    *offspringPopulation[offspringIndex] = elitist;
  }

  return solutionHasChanged;
}

bool PopulationNovelGeneral::conditionalFI(size_t offspringIndex, Individual *backup, vector<vector<int> > &neighbors)
{
  vector<bool> sampled(config->numberOfVariables, false);
  bool solutionHasChanged = 0;

  vector <bool> dependentMonitor(config->numberOfVariables, 0);

  size_t FOSidx = populationFOSIndices[offspringIndex];
  // Locally override the instance FOSInstance.
  FOS *FOSInstance = FOSs[FOSidx];
  
  Individual &elitist = *getElitist();

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
      if (offspringPopulation[offspringIndex]->genotype[dependent[k]] != elitist.genotype[dependent[k]])
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
      offspringPopulation[offspringIndex]->genotype[variableFromFOS] = elitist.genotype[variableFromFOS];
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
    *offspringPopulation[offspringIndex] = elitist;
  }

  return solutionHasChanged;
}

void PopulationNovelGeneral::evaluateSolution(Individual *solution)
{  
  // cout << "Start evaluating...\n"; // DEBUGPRINT
  BitflipLS bitflipLS = BitflipLS(config, sharedInformationPointer, problemInstance);
  
  // Just evaluate.
  vector<int> changedGenes;
  bitflipLS.evaluateSolution(solution, NULL, changedGenes, populationSize);
  
  updateElitistAndCheckVTR(solution);
}

void PopulationNovelGeneral::checkTimeLimit()
{
  if (getMilliSecondsRunningSinceTimeStamp(sharedInformationPointer->startTimeMilliseconds) > config->timelimitSeconds*1000)
  {
    cout << "TIME LIMIT REACHED!" << endl;
    throw customException("time");
  }
}

void PopulationNovelGeneral::updateElitistAndCheckVTR(Individual *solution)
{
  // Global elitist is already updated
  // Update population elitist (or some other elitist)
  Individual* elitist_ptr = getElitist();
  if (!hasElitist() || solution->fitness > elitist_ptr->fitness)
  {
    // Copy.
    setElitist(*solution);
  }
}

int PopulationNovelGeneral::compareSolutions(Individual *x, Individual*y)
{
  if (x->fitness > y->fitness)
    return 1;
  if (x->fitness == y->fitness)
    return 0;
  return -1;
}

size_t PopulationNovelGeneral::getKforKNN(int p)
{
  if (p > 0)
  {
    return p;
  }
  if (p == -1)
  {
    return ceil(sqrt(populationSize));
  }
  if (p == -2)
  {
    return ceil(log2(populationSize));
  }
  if (p == -3)
  {
    size_t min_samples_per_group = std::min(populationSize, (size_t) 16);
    uniform_int_distribution<size_t> d(1, (size_t) ceil(static_cast<double>(populationSize) / min_samples_per_group));
    return ceil(populationSize / d(this->config->rng));
  }

  cerr << "Unknown scheme " << p << " for determining k provided.\n";
  cerr << "Try a positive integer for a fixed value of k.\n";
  cerr << "Or use -1 to use the square root of the population size." << endl;
  exit(0); 
}

void PopulationNovelGeneral::updatePopulationTopology()
{
  vector<double> pg_dist;
  bool isUpdateNeccesary = (this->config->populationTopology != 0);

  if (!isUpdateNeccesary) return;
  upsizeDistanceMatrix();
  population_graph_is_full_graph = false;
  population_cluster_graph_is_full_graph = false;

  switch (this->config->populationTopology)
  {
  // case 0: do nothing (full choice matrix does not change.)
  case 1:
    // Cluster Nearest Better
    updatePopulationDistanceMatrix();
    updatePopulationTopologyNearestBetter(populationGraph, pg_dist);
    prunePopulationGraph(populationGraph, pg_dist);
    clusterifyPopulationGraph(populationGraph, &clusterIndices);
    clusterGraphEqualsPopulationGraph();
    break;
  case 2:
    // Nearest Better Tree
    updatePopulationDistanceMatrix();
    updatePopulationTopologyNearestBetter(populationGraph, pg_dist);
    prunePopulationGraph(populationGraph, pg_dist);
    break;
  case 3:
    // Cluster MST
    updatePopulationDistanceMatrix();
    updatePopulationTopologyMST(populationGraph, pg_dist);
    prunePopulationGraph(populationGraph, pg_dist);
    clusterifyPopulationGraph(populationGraph, &clusterIndices);
    clusterGraphEqualsPopulationGraph();
    break;
  case 4:
    // MST
    updatePopulationDistanceMatrix();
    updatePopulationTopologyMST(populationGraph, pg_dist);
    prunePopulationGraph(populationGraph, pg_dist);
    clusterGraphEqualsPopulationGraph();
    break;
  case 5:
    // Full Graph Prune. Not neccesarily useful.
    updatePopulationDistanceMatrix();
    updatePopulationTopologyFull(populationGraph, pg_dist);
    prunePopulationGraph(populationGraph, pg_dist);
    clusterGraphEqualsPopulationGraph();
    break;
  case 6:
    // KNN, assymmetric
    updatePopulationDistanceMatrix();
    updatePopulationTopologyKNN(populationGraph, &clusterIndices, pg_dist, false, false, false, 0, getKforKNN(config->topologyMode));
    clusterGraphEqualsPopulationGraph();
    break;
  case 7:
    // KNN, symmetric
    {
      int k_knn = getKforKNN(config->topologyMode);
      updatePopulationDistanceMatrix();
      updatePopulationTopologyKNN(populationGraph, &clusterIndices, pg_dist, true, false, false, 0, k_knn);
      clusterGraphEqualsPopulationGraph();
      break;
    }
  case 8:
    // KNN, assymmetric, direction is from better fitness
    // eg. if B is a k-NN of A, normally A->B
    // If A is better than B, the direction is reversed: B->A
    updatePopulationDistanceMatrix();
    updatePopulationTopologyKNN(populationGraph, &clusterIndices, pg_dist, false, true, false, 0, getKforKNN(config->topologyMode));
    clusterGraphEqualsPopulationGraph();
    break;
  case 9:
    // KNN Symmetric & Better Filter
    updatePopulationDistanceMatrix();
    updatePopulationTopologyKNN(populationGraph, &clusterIndices, pg_dist, true, true, false, 1, getKforKNN(config->topologyMode));
    clusterGraphEqualsPopulationGraph();
    break;
  case 10:
    // KNN Assymmetric & Better Filter
    updatePopulationDistanceMatrix();
    updatePopulationTopologyKNN(populationGraph, &clusterIndices, pg_dist, false, true, false, 1, getKforKNN(config->topologyMode));
    clusterGraphEqualsPopulationGraph();
    break;
  case 11:
    // KNN Symmetric & Strictly Better Filter
    updatePopulationDistanceMatrix();
    updatePopulationTopologyKNN(populationGraph, &clusterIndices, pg_dist, true, true, false, 2, getKforKNN(config->topologyMode));
    clusterGraphEqualsPopulationGraph();
    break;
  case 12:
    // KNN Assymmetric & Strictly Better Filter
    updatePopulationDistanceMatrix();
    updatePopulationTopologyKNN(populationGraph, &clusterIndices, pg_dist, false, true, false, 2, getKforKNN(config->topologyMode));
    clusterGraphEqualsPopulationGraph();
    break;
  case 13:
    // Leader Clustering
    // No updatePopulationDistanceMatrix required! (leaderCluster computes distances on-the-fly)
    leaderCluster(populationGraph, &clusterIndices, config->topologyThreshold);
    clusterGraphEqualsPopulationGraph();
    break;
  case 14:
    // Farthest-Nearest Clustering / Greedy Subset Scattering
    // Take a point, and calculate distance to other points.
    // For each point in which the 
    // No updatePopulationDistanceMatrix required! (leaderCluster computes distances on-the-fly)
    greedySubsetScattering(populationGraph, &clusterIndices, getKforGreedySubsetScattering(config->topologyMode));
    clusterGraphEqualsPopulationGraph();
    break;
  case 15:
    // Balanced K-Leader Means Clustering for Linkage Learning
    // KNN for Mating Restriction
    {
    int k_scatter = getKforGreedySubsetScattering(config->topologyMode);
    int k_knn = min((2 * populationSize) / k_scatter, populationSize);
    greedySubsetScattering(populationClusterGraph, &clusterIndices, k_scatter);
    updateClusterContentKNN(populationClusterGraph, clusterIndices, pg_dist, false, false, false, 0, k_knn);
    updatePopulationDistanceMatrix();
    updatePopulationTopologyKNN(populationGraph, NULL, pg_dist, true, false, false, 0, k_knn);
    // cout << "# of clusters: " << clusterIndices.size() << endl;
    }
    break;
  case 16:
    // Balanced K-Leader Means Clustering for Linkage Learning
    // Greedy Subset Scattering followed by KNN
    // For overlapping clusters obtained quickly.
    {
      int k_scatter = getKforGreedySubsetScattering(config->topologyMode);
      int k_knn = min((2 * populationSize) / k_scatter, populationSize);
      greedySubsetScattering(populationGraph, &clusterIndices, k_scatter);
      updateClusterContentKNN(populationClusterGraph, clusterIndices, pg_dist, false, false, false, 0, k_knn);
    }
    break;
  case 17:
    // Greedy Subset Scattering followed by KNN
    // For overlapping clusters obtained quickly.
    {
      int k_scatter = getKforGreedySubsetScattering(config->topologyMode);
      // int k_knn = ceil(log2(populationSize) * 4) + 1;
      int k_knn = min((2 * populationSize) / k_scatter, populationSize);
      greedySubsetScattering(populationGraph, &clusterIndices, k_scatter);
      updateClusterContentKNN(populationClusterGraph, clusterIndices, pg_dist, false, false, false, 0, k_knn);
      populationGraphEqualsClustersClusterGraph();
    }
    break;
  case 18:
    // This should allow recombination over the full population.
    {
      int k_scatter = getKforGreedySubsetScattering(config->topologyMode);
      int k_knn = min((2 * populationSize) / k_scatter, populationSize);
      greedySubsetScattering(populationClusterGraph, &clusterIndices, k_scatter);
      updateClusterContentKNN(populationClusterGraph, clusterIndices, pg_dist, false, false, false, 0, k_knn);
    }
    break;
  case 19:
    // KNN, symmetric
    {
      int k_knn = getKforKNN(config->topologyMode);
      updatePopulationDistanceMatrix();
      updatePopulationTopologyKNN(populationGraph, &clusterIndices, pg_dist, true, false, false, 3, k_knn);
      clusterGraphEqualsPopulationGraph();
      break;
    }
  default:
    break;
  }
  
  if (config->verbosity >= 2)
  {
    cout << "Dumping solution neighbors generated by approach " << this->config->populationTopology << endl;
    for (size_t i = 0; i < populationSize; ++i)
    {
      cout << "[" << i << "] (" << population[i]->fitness << "):";
      for (auto p: getPopulationGraphForIndex(i))
        cout << p << ",";
      cout << endl;
    }
  }
}

double PopulationNovelGeneral::computeDistance(const size_t kind, Individual *a, Individual *b)
{
  const size_t genotypeSize = this->config->numberOfVariables;
  
  switch (kind)
  {
  case 0:
    // Hamming Distance
    {
      double result = 0.0;
      for (size_t i = 0; i < genotypeSize; ++i)
      {
        result += a->genotype[i] != b->genotype[i];
      }

      return result;
    }
    break;
  case 1:
    // Fancy MI distance, is normally a similarity score.
    {
      double result = 0.0;
      for (size_t i = 0; i < genotypeSize; ++i)
      {
        result += a->genotype[i] != b->genotype[i];
      }

      double inv_result = genotypeSize - result;

      return genotypeSize * genotypeSize - (result * result + inv_result * inv_result);
    }
  case 2:
    // Normalized Hamming Distance
    {
      double result = 0.0;
      for (size_t i = 0; i < genotypeSize; ++i)
      {
        result += a->genotype[i] != b->genotype[i];
      }

      return result / genotypeSize;
    }
  case 3:
    // Normalized Fancy MI Distance
    {
      double result = 0.0;
      for (size_t i = 0; i < genotypeSize; ++i)
      {
        result += a->genotype[i] != b->genotype[i];
      }

      result /= genotypeSize;

      double inv_result = 1 - result;

      return 1 - (result * result + inv_result * inv_result);
    }
  case 4:
  // Inverse Entropy weighted distance.
  {
    double eps = 0.1;
    if (lastUpdateOfUnivariateEntropies != numberOfGenerations) {
      // Update entropies.
      // Reset (use as counter)
      univariateEntropies.resize(config->numberOfVariables);
      for (size_t i = 0; i < config->numberOfVariables; ++i)
      {
        univariateEntropies[i] = 0.0;
      }
      // Count
      for (size_t p = 0; p < populationSize; ++p) {
        for (size_t i = 0; i < config->numberOfVariables; ++i)
        {
          // Note: Assuming binary...
          univariateEntropies[i] += population[p]->genotype[i];
        }
      }
      // Compute entropy
      sumOfUnivariateEntropies = 0.0;
      cout << "Computing variable entropies:\n";
      for (size_t i = 0; i < config->numberOfVariables; ++i)
      {
        // p
        double p = univariateEntropies[i] / populationSize;
        double e = 0.0;
        if (p > 0.0 && p < 1.0) e = -p * log2(p) - (1 - p) * log2(1 - p);
        sumOfUnivariateEntropies += e;
        univariateEntropies[i] = e;
        cout << e;
        if (i == config->numberOfVariables - 1) {
          cout << endl;
        } else {
          cout << ",";
        }
      }
      lastUpdateOfUnivariateEntropies = numberOfGenerations;
    }
    // Inv-Entropy-Weighted Normalized Hamming Distance
    {
      double result = 0.0;
      for (size_t i = 0; i < genotypeSize; ++i)
      {
        result += (eps + 1 - univariateEntropies[i]) * (a->genotype[i] != b->genotype[i]);
      }

      return result / ((1 + eps) * config->numberOfVariables - sumOfUnivariateEntropies);
    }
    break;
  }
  case 5:
  // Entropy reduction distance.
  {
    double eps = 0.05;
    if (lastUpdateOfUnivariateEntropies != numberOfGenerations) {
      // Update entropies.
      // Reset (use as counter)
      univariateEntropies.resize(config->numberOfVariables);
      for (size_t i = 0; i < config->numberOfVariables; ++i)
      {
        univariateEntropies[i] = 0.0;
      }
      // Count
      for (size_t p = 0; p < populationSize; ++p) {
        for (size_t i = 0; i < config->numberOfVariables; ++i)
        {
          // Note: Assuming binary...
          univariateEntropies[i] += population[p]->genotype[i];
        }
      }
      // Compute entropy
      sumOfUnivariateEntropies = 0.0;
      cout << "Computing variable entropies:\n";
      for (size_t i = 0; i < config->numberOfVariables; ++i)
      {
        // Probability of seeing a 1.
        double p = univariateEntropies[i] / populationSize;
        double e = 0.0;
        if (p > 0.0 && p < 1.0) e = -p * log2(p) - (1 - p) * log2(1 - p);
        sumOfUnivariateEntropies += e;
        univariateEntropies[i] = p;
        // univariateEntropies[i] = e;
        cout << e;
        if (i == config->numberOfVariables - 1) {
          cout << endl;
        } else {
          cout << ",";
        }
      }
      lastUpdateOfUnivariateEntropies = numberOfGenerations;
    }
    // 
    {
      double result = 0.0;
      for (size_t i = 0; i < genotypeSize; ++i)
      {
        double wa = (a->genotype[i] * univariateEntropies[i]) + ((1 - a->genotype[i]) * (1 - univariateEntropies[i])) + eps;
        double wb = (a->genotype[i] * univariateEntropies[i]) + ((1 - a->genotype[i]) * (1 - univariateEntropies[i])) + eps;
        result += (wb / wa) * (a->genotype[i] != b->genotype[i]);
      }

      return result / ((1 + eps) * config->numberOfVariables - sumOfUnivariateEntropies);
    }

  }

  default:
    cerr << "Unknown distance measure used." << endl;
    exit(0);
    break;
  }
}


void PopulationNovelGeneral::updatePopulationDistanceMatrix()
{
  const size_t genotypeSize = this->config->numberOfVariables;
  const size_t distance_kind = this->config->distance_kind;

  for (size_t i = 0; i < populationSize; ++i)
  {
    for (size_t j = i + 1; j < populationSize; ++j)
    {
      double dist = computeDistance(distance_kind, population[i], population[j]);
      this->populationDistances[i][j] = dist;
      this->populationDistances[j][i] = dist;
    }
  }
}

void PopulationNovelGeneral::leaderCluster(vector<vector<int>> &populationGraph, vector<size_t> *clusterIndices, const double distance_threshold)
{
  const size_t distance_kind = this->config->distance_kind;
  // Clear the current cluster indices.
  if (clusterIndices != NULL)
  {
    clusterIndices->clear();
  }
  vector<size_t> w(populationSize);
  iota(w.begin(), w.end(), 0);

  // Remove the old edges in the population graph.
  for (auto i: w)
  {
    populationGraph[i].clear();
  }

  // Order by fitness: leader is the most fit of its niche.
  // Also keep in mind genotype if fitnesses are equal,
  // such that equal solutions follow one another.
  sort(w.begin(), w.end(), [this](size_t a, size_t b){
    Individual *x = population[a];
    Individual *y = population[b];
    
    if (x->fitness > y->fitness)
      return false;
    else if (x->fitness < y->fitness)
      return true;
    else if (x->genotype > y->genotype)
      return false;
    else if (x->genotype < y->genotype)
      return true;
    else
      return false;
  });
  // shuffle(w.begin(), w.end(), config->rng);

  while (w.size() > 0)
  {
    // Note: the list w is in reverse!
    size_t leader = w.back();
    vector<size_t> cluster;
    if (clusterIndices != NULL)
    {
      clusterIndices->push_back(leader);
      if (clusterIndices != NULL && config->multiFOS >= 2 )
      {
        populationFOSIndices[leader] = leader;
      }
    }
    cluster.push_back(leader);
    w.pop_back();

    auto it = remove_if(w.begin(), w.end(), 
    [this, distance_threshold, distance_kind, &cluster, leader, clusterIndices](size_t a) {
      double distance = computeDistance(distance_kind, population[leader], population[a]);
      bool in_leaders_cluster = distance <= distance_threshold;
      if (in_leaders_cluster)
      {
        if (clusterIndices != NULL && config->multiFOS >= 2 )
        {
          populationFOSIndices[a] = leader;
        }
        cluster.push_back(a);
      }
      return in_leaders_cluster;
    });
    // Actually resize the vector to remove items.
    // remove_if is iterator based and cannot actually do that itself...
    w.resize(std::distance(w.begin(), it));

    // As in the previous setting, update the population graph.
    // Note: This is quadratic in the cluster size(!), it might be good
    // to ignore self-as-donor in GOM itself, as we only need a single vector
    // in that case.
    // Add in all the edges (within the cluster)
    for (auto i: cluster)
    {
      for (auto j: cluster)
      {
        if (i == j) continue;
        populationGraph[i].push_back(j);
      }
    }
  }

}

size_t PopulationNovelGeneral::getKforGreedySubsetScattering(int p)
{
  if (p > 0)
  {
    return p;
  }
  if (p == -1)
  {
    return floor(sqrt(populationSize));
  }
  if (p == -2)
  {
    return floor(log2(populationSize));
  }
  if (p == -3)
  {
    return floor(populationSize / log2(populationSize));
  }
  cerr << "Unknown configuration for number of clusters for FNC." << endl;
  exit(0);
}

void PopulationNovelGeneral::greedySubsetScattering(vector<vector<int>> &populationGraph, vector<size_t> *clusterIndices, const int k)
{
  const size_t distance_kind = this->config->distance_kind;
  // Clear the current cluster indices.
  if (clusterIndices != NULL)
  {
    clusterIndices->clear();
  }

  vector<size_t> w(populationSize);
  iota(w.begin(), w.end(), 0);

  vector<double> d(populationSize);
  fill(w.begin(), w.end(), 0.0);

  // Remove the old edges in the population graph.
  for (size_t i = 0; i < populationSize; ++i)
  {
    populationGraph[i].clear();
  }

  // Starting point is the fittest solution.
  size_t leader = *max_element(w.begin(), w.end(), 
  [this](size_t a, size_t b) {
    Individual *x = population[a];
    Individual *y = population[b];
    
    if (x->fitness > y->fitness)
      return false;
    else if (x->fitness < y->fitness)
      return true;
    else if (x->genotype > y->genotype)
      return false;
    else if (x->genotype < y->genotype)
      return true;
    else
      return false;
  });
  // uniform_int_distribution random_index((size_t) 0, w.size());
  // size_t leader = random_index(config->rng);
  
  //
  size_t next_leader = 0;
  double next_leader_distance = 0.0;
  fill(w.begin(), w.end(), leader);
  for (size_t a = 0; a < populationSize; ++a)
  {
    double da = computeDistance(distance_kind, population[leader], population[a]);
    d[a] = da;
    if (next_leader_distance < da)
    {
      next_leader = a;
      next_leader_distance = da;
    }
  }

  for (size_t cluster_idx = 1; cluster_idx < k; ++cluster_idx)
  {
    leader = next_leader;
    next_leader_distance = 0.0;
    for (size_t a = 0; a < populationSize; ++a)
    {
      double da = computeDistance(distance_kind, population[leader], population[a]);
      populationDistances[leader][a] = da;
      populationDistances[a][leader] = da;
      if (da < d[a])
      {
        d[a] = da;
        w[a] = leader;

        if (clusterIndices != NULL && config->multiFOS >= 2)
        {
          populationFOSIndices[a] = leader;
        }
      } else {
        da = d[a];
      }
      if (next_leader_distance < d[a])
      {
        next_leader = a;
        next_leader_distance = da;
      }
    }
  }
  
  // At this point every solution has its cluster assignment in w.
  // Turn into population graph.

  vector<vector<size_t>> clusters(populationSize);
  for (size_t i = 0; i < populationSize; ++i)
  {
    clusters[w[i]].push_back(i);
  }

  for (size_t cluster_idx = 0; cluster_idx < populationSize; ++cluster_idx)
  {
    auto cluster = clusters[cluster_idx];
    for (auto i: cluster)
    {
      for (auto j: cluster)
      {
        if (i == j) continue;
        populationGraph[i].push_back(j);
      }
    }
    if (clusterIndices != NULL && cluster.size() > 0)
    {
      clusterIndices->push_back(cluster_idx);
    }
  }

}

void PopulationNovelGeneral::updateClusterContentKNN(
  vector<vector<int>> &populationGraph,
  vector<size_t> &clusterIndices,
  vector<double> &distances,
  const bool symmetric,
  const bool fitness_direction,
  const bool reverse,
  const int filter,
  const size_t k
  )
{
  const int distance_mul = reverse ? -1 : 1;

  vector<vector<bool>> neighbors(populationSize);
  for (size_t i = 0; i < populationSize; ++i)
  {
    populationGraph[i].clear();
    
    neighbors[i].resize(populationSize);
    fill(neighbors[i].begin(), neighbors[i].end(), false);
  }

  vector<pair<double, size_t>> knn;

  for (auto i: clusterIndices)
  {
    knn.clear();
    for (size_t j = 0; j < populationSize; ++j)
    {
      if (i == j) continue;
      // Filter type 1: do not account for worse solutions in the KNN.
      // Basically a hybrid of KNN & Nearest Better.
      if (filter == 1 && population[j]->fitness < population[i]->fitness) continue;
      if (filter == 2 && population[j]->fitness <= population[i]->fitness) continue;
      auto pair_distance = distance_mul * populationDistances[i][j];
      if (filter == 3 && pair_distance <= 0.0) continue;
      knn.push_back(pair(pair_distance, j));
    }
    // Find the k-th nearest neighbors. (unsorted)
    if (k < knn.size())
      nth_element(knn.begin(), knn.begin() + k - 1, knn.end());
    
    if (config->verbosity >= 2)
    {
      cout << "K = " << k << "; Distances: ";
      size_t k_left = k;
      bool first = true;
      for (auto nn: knn)
      {
        if (!first) cout << ",";
        first = false;
        cout << nn.first;
        if (k_left == 1)
        {
          cout << "|";
        }   

        if(k_left > 0) --k_left;
      }
      cout << endl;
    }

    knn.resize(min(k, knn.size()));
    // Set up the Population Graph
    for (pair<double, size_t> knnel: knn)
    {
      size_t j = knnel.second;
      if (symmetric)
      {
        if (!neighbors[i][j])
        {
          populationGraph[i].push_back(j);
          populationGraph[j].push_back(i);
          neighbors[i][j] = true;
          neighbors[j][i] = true;
        }
      }
      else
      {
        if (fitness_direction && population[i]->fitness >= population[j]->fitness)
        {
          if (!neighbors[j][i])
          {
            populationGraph[j].push_back(i);
            neighbors[j][i] = true;
          }
        }
        else
        {
          if (!neighbors[i][j]) 
          {
            populationGraph[i].push_back(j);
            neighbors[i][j] = true;
          }
        }
        
      }
    }
  }
}

void PopulationNovelGeneral::updatePopulationTopologyFull(vector<vector<int>> &populationGraph, vector<double> &pg_dist)
{
  pg_dist.clear();
  for (size_t i = 0; i < populationSize; ++i)
  {
    populationGraph[i].resize(populationSize);
    iota(populationGraph[i].begin(), populationGraph[i].end(), 0);
    for (size_t j = i + 1; j < populationSize; ++j)
    {
      pg_dist.push_back(populationDistances[i][j]);
    }
  }
}


void PopulationNovelGeneral::updatePopulationTopologyNearestBetter(vector<vector<int>> &populationGraph, vector<double> &pg_dist)
{
  // Assume distances have been recalculated.

  // Clear previous graph
  for (size_t i = 0; i < populationSize; ++i)
  {
    populationGraph[i].clear();
  }

  for (size_t i = 0; i < populationSize; ++i)
  {
    double fitness_i = population[i]->fitness;
    auto populationDistances_i = populationDistances[i];

    size_t nearest_better = i;
    double nearest_better_distance = INFINITY;
    bool nearest_better_better = false;

    for (size_t j = 0; j < populationSize; ++j)
    {
      // Skip over self.
      if (i == j) continue;

      double distance = populationDistances[i][j];
      double fitness_j = population[j]->fitness;
      bool is_better = fitness_j > fitness_i;
      
      // Skip if strictly worse
      if (fitness_j < fitness_i) continue;
      // Skip if equal fitness, and we don't allow equal fitness.
      if (fitness_j == fitness_i && !config->topologyParam2) continue;
      // Skip if not better and if nearest_better is better.
      // And we include equal fitness if no better solutions exist.
      if (nearest_better_better && !is_better) continue;
      
      if (distance < nearest_better_distance)
      {
        nearest_better = j;
        nearest_better_distance = distance;
        nearest_better_better = is_better;
      }
    }

    if (nearest_better != i &&
       (nearest_better_better || i > nearest_better))
    {
      // Nearest better found!
      populationGraph[i].push_back(nearest_better);
      populationGraph[nearest_better].push_back(i);
      pg_dist.push_back(nearest_better_distance);
    }

  }

  // At this point we have the full nearest better tree in populationGraph.

  if (config->verbosity >= 2)
  {
    cout << "Distances in Nearest Better Tree:" << endl;
    for (double d: pg_dist)
    {
      cout << d << ",";
    }
    cout << endl;
  }

}

void PopulationNovelGeneral::updatePopulationTopologyMST(vector<vector<int>> &populationGraph, vector<double> &pg_dist)
{
  // Clear previous graph
  for (auto g: populationGraph)
  {
    g.clear();
  }

  // Note to self: technically O(n^2 log n)
  // So the NN-Chain algorithm may be faster...
  // Perform Kruskal's Algorithm
  vector<size_t> unionFind(populationSize);
  iota(unionFind.begin(), unionFind.end(), 0);
  vector<size_t> unionFindSizes(populationSize);
  fill(unionFindSizes.begin(), unionFindSizes.end(), 1);
  vector<size_t> chain;

  size_t num_self_reprs = populationSize;

  auto findSetRepr = [&unionFind, &chain](size_t i)
  {
    // Find union
    chain.clear();
    chain.push_back(i);
    while (true)
    {
      size_t current = chain.back();
      size_t next = unionFind[current];
      if (current == next) 
      {
        break;
      }
      chain.push_back(next);
    }

    // Compress
    size_t representative = chain.back();
    for (size_t i: chain)
    {
      unionFind[i] = representative;
    }

    return chain.back();
  };

  // Build priority queue
  vector<tuple<double, size_t, size_t>> pq;
  pq.reserve(populationSize * populationSize);
  for (size_t i = 0; i < populationSize; ++i)
  {
    for (size_t j = i+1; j < populationSize; ++j)
    {
      pq.push_back(tuple(-populationDistances[i][j], i, j));
    }
  }

  make_heap(pq.begin(), pq.end());

  // Until all sets are merged (eg. have representative 0)
  // or until we are out of edges.
  while (num_self_reprs > 1)
  {

    tuple<double, size_t, size_t> edge = pq.front();
    pop_heap(pq.begin(), pq.end());
    pq.resize(pq.size() - 1);

    double distance = -get<0>(edge);
    size_t a = get<1>(edge);
    size_t b = get<2>(edge);

    // cout << "(1) Currently there are " << num_zero_reprs << " items with zero as representative" << endl;
    // cout << "Got edge: " << a << "<->" << b << " (" << distance << ")" << endl;

    size_t repr_a = findSetRepr(a);
    size_t repr_b = findSetRepr(b);

    // cout << "a (" << a << ") belongs to " << repr_a << endl;
    // cout << "b (" << b << ") belongs to " << repr_b << endl;

    // cout << "(2) Currently there are " << num_zero_reprs << " items with zero as representative" << endl;

    if (repr_a != repr_b)
    {
      // Add edge!
      populationGraph[a].push_back(b);
      populationGraph[b].push_back(a);

      pg_dist.push_back(distance);

      // Merge sets
      // Note: more traditionally a size array is kept, and the representative
      // with the largest size is chosen.
      size_t new_repr = unionFindSizes[repr_a] > unionFindSizes[repr_b] ? repr_a : repr_b;
      size_t old_repr = unionFindSizes[repr_a] > unionFindSizes[repr_b] ? repr_b : repr_a;

      unionFind[old_repr] = new_repr;
      unionFindSizes[new_repr] += unionFindSizes[old_repr];
      num_self_reprs -= 1;
      
    }

  }

  if (config->verbosity >= 2)
  {
    cout << "Distances in MST:" << endl;
    for (double d: pg_dist)
    {
      cout << d << ",";
    }
    cout << endl;
  }
}

void PopulationNovelGeneral::updatePopulationTopologyKNN(
  vector<vector<int>> &populationGraph,
  vector<size_t> *clusterIndices,
  vector<double> &distances,
  const bool symmetric,
  const bool fitness_direction,
  const bool reverse,
  const int filter,
  const size_t k
  )
{
  const int distance_mul = reverse ? -1 : 1;

  vector<vector<bool>> neighbors(populationSize);
  for (size_t i = 0; i < populationSize; ++i)
  {
    populationGraph[i].clear();
    
    neighbors[i].resize(populationSize);
    fill(neighbors[i].begin(), neighbors[i].end(), false);
  }

  vector<pair<double, size_t>> knn;

  for (size_t i = 0; i < populationSize; ++i)
  {
    knn.clear();
    for (size_t j = 0; j < populationSize; ++j)
    {
      if (i == j) continue;
      // Filter type 1: do not account for worse solutions in the KNN.
      // Basically a hybrid of KNN & Nearest Better.
      if (filter == 1 && population[j]->fitness < population[i]->fitness) continue;
      if (filter == 2 && population[j]->fitness <= population[i]->fitness) continue;
      auto pair_distance = distance_mul * populationDistances[i][j];
      if (filter == 3 && pair_distance <= 0.0) continue;
      knn.push_back(pair(pair_distance, j));
    }
    // Find the k-th nearest neighbors. (unsorted)
    if (k < knn.size())
      nth_element(knn.begin(), knn.begin() + k - 1, knn.end());
    
    if (config->verbosity >= 2)
    {
      cout << "K = " << k << "; Distances: ";
      size_t k_left = k;
      bool first = true;
      for (auto nn: knn)
      {
        if (!first) cout << ",";
        first = false;
        cout << nn.first;
        if (k_left == 1)
        {
          cout << "|";
        }   

        if(k_left > 0) --k_left;
      }
      cout << endl;
    }

    knn.resize(min(k, knn.size()));
    // Set up the Population Graph
    for (pair<double, size_t> knnel: knn)
    {
      size_t j = knnel.second;
      if (symmetric)
      {
        if (!neighbors[i][j])
        {
          populationGraph[i].push_back(j);
          populationGraph[j].push_back(i);
          neighbors[i][j] = true;
          neighbors[j][i] = true;
        }
      }
      else
      {
        if (fitness_direction && population[i]->fitness >= population[j]->fitness)
        {
          if (!neighbors[j][i])
          {
            populationGraph[j].push_back(i);
            neighbors[j][i] = true;
          }
        }
        else
        {
          if (!neighbors[i][j]) 
          {
            populationGraph[i].push_back(j);
            neighbors[i][j] = true;
          }
        }
        
      }
    }
  }

  // Each vertex is its own cluster.
  if (clusterIndices != NULL)
  {
    clusterIndices->resize(populationSize);
    iota(clusterIndices->begin(), clusterIndices->end(), 0);
  }
}

void PopulationNovelGeneral::prunePopulationGraph(vector<vector<int>> &populationGraph, vector<double> distances)
{
  // Now perform filtering, depending on setting.
  // Note: It does not make much sense to NOT filter when clusterify-ing afterwards.
  //       as the result will always be a fully connected graph.
  //       In that case you are better off setting populationTopology to 0,
  //       and save yourself some computational cycles.
  switch (config->topologyMode)
  {
  case 0:
    prunePopulationGraphDistanceThreshold(populationGraph, distances);
    break;
  case 1:
    prunePopulationGraphDistanceThresholdInterquartile(populationGraph, distances);
    break;
  case 2:
    prunePopulationGraphRelink(populationGraph);
    break;
  default:
    break;
  }
}

void PopulationNovelGeneral::prunePopulationGraphDistanceThreshold(vector<vector<int>> &populationGraph, vector<double> distances)
{
  // No distances: nothing to remove!
  if (distances.size() == 0)
    return;
  
  // Calculate threshold by rank:
  size_t r = (size_t) (config->topologyThreshold * ((double) distances.size()));
  nth_element(distances.begin(), distances.begin() + r, distances.end());
  double max_dist = distances[r];
  
  // Filter by distances.
  for (size_t i = 0; i < populationSize; ++i)
  {
    const auto d = this->populationDistances;
    auto it = remove_if(populationGraph[i].begin(), populationGraph[i].end(), 
      [i, d, max_dist](int b) {
        return d[i][b] > max_dist;
      });
    populationGraph[i].resize(distance(populationGraph[i].begin(), it));
  }
}

void PopulationNovelGeneral::prunePopulationGraphDistanceThresholdInterquartile(vector<vector<int>> &populationGraph, vector<double> distances)
{
  // No distances: nothing to remove!
  if (distances.size() == 0)
    return;
  
  // Calculate threshold by rank:
  size_t qr25 = distances.size() / 4;
  size_t qr75 = distances.size() - distances.size() / 4;
  auto q25 = distances.begin() + qr25;
  auto q75 = distances.begin() + qr75;

  nth_element(distances.begin(), q25, distances.end());
  nth_element(q25, q75, distances.end());

  double iqr = *q75 - *q25;
  double max_dist = *q75 + 1.5 * iqr;
  
  // Filter by distances.
  for (size_t i = 0; i < populationSize; ++i)
  {
    const auto d = this->populationDistances;
    auto it = remove_if(populationGraph[i].begin(), populationGraph[i].end(), 
      [i, d, max_dist](int b) {
        return d[i][b] > max_dist;
      });
    populationGraph[i].resize(distance(populationGraph[i].begin(), it));
  }
}

void PopulationNovelGeneral::prunePopulationGraphRelink(vector<vector<int>> &populationGraph)
{
  Linkup linkup = Linkup(config, sharedInformationPointer, problemInstance);
  
  vector<vector<bool>> relinkable(populationSize);
  for (size_t i = 0; i < populationSize; ++i)
  {
    relinkable[i].resize(populationSize);
  }

  size_t failures = 0;
  size_t successes = 0;

  // Compute which edges can be relinked.
  for (size_t i = 0; i < populationSize; ++i)
  {
    for (size_t j: populationGraph[i])
    {
      // Don't compute the same thing twice.
      if (j <= i) continue; 
      
      bool success = linkup.relink(population[i], population[j]);
      relinkable[i][j] = success;
      relinkable[j][i] = success;

      failures += !success;
      successes += success;
    }
  }
  
  // Printing the relinking statistics.
  if (config->verbosity >= 2)
  {
    cout << "Pruning by relinking. " << successes << " successes, " << failures << " failures." << endl;
    cout << "Successes: ";
    for (size_t i = 0; i < populationSize; ++i)
    {
      for (size_t j: populationGraph[i])
      {
        if (j <= i) continue;
        if (relinkable[i][j])
        {
          cout << "(" << i << " [" << population[i]-> fitness << "], " << j << " [" << population[j]->fitness << "])[" << populationDistances[i][j] << "]; ";
        }
      }
    }
    cout << endl;
    cout << "Failures: ";
    for (size_t i = 0; i < populationSize; ++i)
    {
      for (size_t j: populationGraph[i])
      {
        if (j <= i) continue;
        if (!relinkable[i][j])
        {
          cout << "(" << i << " [" << population[i]-> fitness << "], " << j << " [" << population[j]->fitness << "])[" << populationDistances[i][j] << "]; ";
        }
      }
    }
    cout << endl;
  }
  // Prune edges.
  for (size_t i = 0; i < populationSize; ++i)
  {
    auto r = remove_if(populationGraph[i].begin(), populationGraph[i].end(),
      [&relinkable, i](size_t j) {
        return !relinkable[i][j];
      });
    populationGraph[i].erase(r, populationGraph[i].end());
  }

}

void PopulationNovelGeneral::clusterifyPopulationGraph(vector<vector<int>> &populationGraph, vector<size_t> *clusterIndices)
{
  // The population graph at this point consists of connected components
  // That are not fully connected themselves.
  // This routine turns every connected component into a fully connected component.

  // Clear the current clusters
  if (clusterIndices != NULL)
  {
    clusterIndices->clear();
  }
  // Keep track of which elements we have seen before.
  vector<bool> visited(populationSize);
  fill(visited.begin(), visited.end(), false);

  vector<size_t> unvisited(populationSize);
  iota(unvisited.begin(), unvisited.end(), 0);

  // Queue of elements we are currently visiting.
  queue<size_t> queue;
  // Set of items seen in the currently connected component so far.
  vector<size_t> cluster;

  while (!unvisited.empty())
  {
    size_t start = unvisited.back();
    unvisited.pop_back();

    // Skip items that have already been visited.
    if (visited[start]) continue;
    
    // Start a new cluster
    cluster.clear();
    // Start visiting the connected component off with start.
    queue.push(start);
    visited[start] = true;
    cluster.push_back(start);
    
    while(!queue.empty())
    {
      size_t current = queue.front();
      queue.pop();

      for (auto neighbor: populationGraph[current])
      {
        // Skip over already visited nodes.
        if (visited[neighbor]) continue;
        // Mark as seen
        visited[neighbor] = true;
        // Add new ones to the queue.
        queue.push(neighbor);
        // Add to cluster
        cluster.push_back(neighbor);
      }
      
    }

    // Cluster should now contain all elements within the connected component.
    // Remove the old edges
    for (auto i: cluster)
    {
      populationGraph[i].clear();
    }
    // Add in all the edges (within the cluster)
    for (auto i: cluster)
    {
      for (auto j: cluster)
      {
        if (i == j) continue;
        populationGraph[i].push_back(j);
      }
    }
    if (clusterIndices != NULL)
    {
      clusterIndices->push_back(cluster.front());
    }
  }

  // The graph should now be fully connected.
}

void PopulationNovelGeneral::clusterGraphEqualsPopulationGraph()
{
  if (population_graph_is_full_graph)
  {
    population_cluster_graph_is_full_graph = true;
  }
  else 
  {
    population_cluster_graph_is_full_graph = false;
    this->populationClusterGraph = this->populationGraph;
  }
}

void PopulationNovelGeneral::populationGraphEqualsClustersClusterGraph()
{
  if (population_cluster_graph_is_full_graph)
  {
    population_graph_is_full_graph = true;
  }
  else
  {
    for (size_t i = 0; i < populationSize; ++i)
    {
      populationGraph[i] = populationClusterGraph[populationFOSIndices[i]];
    }
  }
}