#pragma once

#include <cmath>
#include <iostream> 
#include <vector>
using namespace std;

#include "Individual.hpp"
#include "Config.hpp"
#include "shared.hpp"
#include "problems.hpp"
#include "FOS.hpp"
#include "bitflipls.hpp"

class PopulationNovelGeneral
{
public:
	Config *config;
	Problem *problemInstance;
	sharedInformation *sharedInformationPointer;
	size_t populationIndex;
	size_t populationSize;

	Individual populationElitist;
	bool firstEvaluationPopulation;

	bool hasElitist();
	Individual* getElitist();
	void setElitist(Individual& i);

	/* BEGIN: CHANGES FOR MEMORY USAGE REDUCTION */
	// In order to not waste memory (this turned out to be a problem!)
	// we use a boolean to specify that we are indeed using a full population graph.
	bool population_graph_is_full_graph = true;
	bool population_cluster_graph_is_full_graph = true;
	
	// Alter the distance matrix dimensions. Freeing it once we are done.
	// Keeping a |P|^2 matrix around with multiple populations is apparently where madness lies.
	void upsizeDistanceMatrix();
	void downsizeDistanceMatrix();

	// Same for the population graphs.
	void downsizePopulationGraph();
	void downsizePopulationClusterGraph();

	vector<int> getPopulationGraphForIndex(int population_idx);
	vector<int> getPopulationClusterGraphForIndex(int population_idx);

	// Function to be called at the end of a generation to cleanup allocations that can be redone
	// at the next generation to reduce memory usage.
	void endGeneration();
	/* END: CHANGES FOR MEMORY USAGE REDUCTION */

	double minimumCriterion = -INFINITY;
	
	// Keep track of bounds of the fitness function
	double inherentFitnessMin = INFINITY;
	double inherentFitnessMax = -INFINITY;
	// If the distance metric is sane (eg. always positive)
	double inherentNoveltyMin = 0.0;
	double inherentNoveltyMax = 1.0;

	vector<Individual*> population;
	vector<Individual*> offspringPopulation;
	vector<int> noImprovementStretches;

	vector<double> univariateEntropies;
	double sumOfUnivariateEntropies;
	size_t lastUpdateOfUnivariateEntropies = -1;

	vector<vector<double>> populationDistances;
	vector<vector<int>> populationGraph;
	// Similar to populationGraph. A separate graph for
	// model building only.
	// (eg. clustering for models, but KNN for mating restriction)
	vector<vector<int>> populationClusterGraph;

	FOS *populationFOS;
	bool terminated;
	double averageFitness;
	size_t numberOfGenerations;
	FOS *FOSInstance = NULL;
	
	// A vector of FOSes.
	vector<FOS*> FOSs;
	// A vector of FOS indices, eg. assigning an index to each individual to a FOS.
	// Note: May change depending on needs in the future.
	vector<size_t> populationFOSIndices;
	vector<size_t> clusterIndices;

	vector<vector<double> > matrix;

	deque<Individual*> noveltyArchive;

	PopulationNovelGeneral(Config *config_, Problem *problemInstance_, sharedInformation *sharedInformationPointer_, size_t populationIndex_, size_t populationSize_);
	virtual ~PopulationNovelGeneral(){};

	void tournamentSelection(int k, vector<Individual*> &population, vector<Individual*> &offspringPopulation);
	void hillClimberSingle(Individual *solution);	
	void hillClimberMultiple(Individual *solution);

	void calculateAverageFitness();	
	void copyOffspringToPopulation();
	bool GOM(size_t offspringIndex, Individual *backup);
	void findNeighbors(vector<vector< int> > &neighbors);
	bool conditionalGOM(size_t offspringIndex, Individual *backup, vector<vector<int> > &neighbors);	
	bool FI(size_t offspringIndex, Individual *backup);
	bool conditionalFI(size_t offspringIndex, Individual *backup, vector<vector<int> > &neighbors);
	void updateElitistAndCheckVTR(Individual *solution);
	void checkTimeLimit();
	int compareSolutions(Individual *x, Individual*y);
	
	void evaluateSolution(Individual *solution);

	void updatePopulationDistanceMatrix();
	void updatePopulationTopology();

	double computeDistance(const size_t kind, Individual *a, Individual *b);

private:
	void updatePopulationTopologyFull(vector<vector<int>> &populationGraph, vector<double> &pg_dist);
	void updatePopulationTopologyNearestBetter(vector<vector<int>> &populationGraph, vector<double> &pg_dist);
	void updatePopulationTopologyMST(vector<vector<int>> &populationGraph, vector<double> &pg_dist);
	void updatePopulationTopologyKNN(vector<vector<int>> &populationGraph, vector<size_t> *clusterIndices, vector<double> &pg_dist, const bool symmetric, const bool fitness_direction, const bool reverse, const int filter, const size_t k);
	size_t getKforKNN(int p);

	void prunePopulationGraph(vector<vector<int>> &populationGraph, vector<double> distances);
	void prunePopulationGraphDistanceThreshold(vector<vector<int>> &populationGraph, vector<double> distances);
	void prunePopulationGraphDistanceThresholdInterquartile(vector<vector<int>> &populationGraph, vector<double> distances);
	void prunePopulationGraphRelink(vector<vector<int>> &populationGraph);

	void clusterifyPopulationGraph(vector<vector<int>> &populationGraph, vector<size_t> *clusterIndices);
	void leaderCluster(vector<vector<int>> &populationGraph, vector<size_t> *clusterIndices, const double distance_threshold);
	void greedySubsetScattering(vector<vector<int>> &populationGraph, vector<size_t> *clusterIndices, const int k);
	void updateClusterContentKNN(vector<vector<int>> &populationGraph, vector<size_t> &clusterIndices, vector<double> &distances, const bool symmetric, const bool fitness_direction, const bool reverse, const int filter, const size_t k);
	size_t getKforGreedySubsetScattering(int p);

	void clusterGraphEqualsPopulationGraph();
	void populationGraphEqualsClustersClusterGraph();
};

