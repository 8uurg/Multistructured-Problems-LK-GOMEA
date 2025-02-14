//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
//
// This research code was modified as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
//
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt (modifications for relevant article), Arkadiy Dushatskiy (original developer)

#pragma once

#include <optional>
#include <string>
#include <cstdlib>
#include <iostream>
#include <random>

#include <getopt.h>

using namespace std;

#include "problems.hpp"

extern const double INF_FITNESS;

class Config
{
	void splitString(const string &str, vector<string> &splitted, char delim);
	bool isNumber(const string &s);

public:
	int usePartialEvaluations              = 0,                  
	    AnalyzeFOS                         = 0,
	    saveEvaluations                    = 0,
	    useForcedImprovements              = 1,
   		printHelp                          = 0,
		verbosity                          = 0;
	int approach                           = 1;

	int use_elitist_per_population = 0;

	string command;

   	double ratioVIG;
   	int conditionalGOM = 0;
   	double MI_threshold = 0.5;
   	double vtr = 1e+308;
	int vtr_n_unique = 1;
	size_t problemIndex = 0, k = 1, s = 1, fn = 1,  
		FOSIndex = 3,              
	    orderFOS = 0, // Always use random ordering
	    numberOfVariables = 1;
	// 0 - MI
	// 1 - NMI
	// 2 - Random
	int similarityMeasure = 1;

	// What population topology to use
	// 0 - Full
	// 1 - Nearest Better Clustering (distance threshold)
	// 2 - Nearest Better Topology
	// 3 - MST Cluster
	// 4 - MST
	// 5 - Pruned Full Graph
	// 6 - KNN, assymetric (k)
	// 7 - KNN, symmetric (k)
	// 8 - KNN, assymetric -- neighbor is in direction of improvement (k)
	// 9 - KNN, symmetric & better filter (k)
	// 10 - KNN, assymetric & better filter (k)
	// 11 - KNN, symmetric & strictly better filter (k)
	// 12 - KNN, assymetric & strictly better filter (k)
	// 13 - Leader Clustering (distance threshold)
	int populationTopology = 0;

	// Whether to learn multiple FOSs.
	// 0 - None, single FOS per population.
	// 1 - FOS per Individual.
	int multiFOS = 0;

	// What distance metric to use.
	// 0 - Genotypic Hamming Distance
	// 1 - MI Maximization Similarity
	size_t distance_kind = 0;

	// TODO: What mode should the learner be in
	// - For full: Ignored
	// - For Nearest Better *:
	//   <-1> None (No filter is applied)
	//    <0> Quantile Threshold
	//    <1> Interquartile Distance Threshold
	int topologyMode = 0;

	double topologyThreshold = 0.5;

	int topologyParam2 = 0;

	optional<vector<char>> reference_solution;
	double p_sample_reference = 0.0;


	// What action to perform during GOM
	// 0 = copy block
	// 1 = flip block
	// 2 = copy or flip (/w probability)
	int mixingOperator = 0;

	double flipProbability = 0.05;

	// What fitness sharing method to use
	// Note: the approach is adapted to fit within GOMEA
	//  0 = No fitness sharing
	//  1 = Fitness Sharing -- genotype (absolute)
	//  2 = Fitness Sharing -- genotype (absolute), exclude parent
	//  3 = Fitness Sharing -- genotype (normalized distance)
	//  4 = Fitness Sharing -- genotype (normalized distance), exclude parent
	//  5 = Fitness Sharing -- genotype (absolute), normalize fitness to population
	//  6 = Fitness Sharing -- genotype (absolute), exclude parent, normalize fitness to population
	//  7 = Fitness Sharing -- genotype (normalized distance), normalize fitness to population
	//  8 = Fitness Sharing -- genotype (normalized distance), exclude parent, normalize fitness to population
	int fitness_sharing = 0;

	// Parameters for fitness sharing
	double share_sigma = 7;
	double share_alpha = 5;

	bool share_fitness_filter = false;
	bool share_steadystate = true;
	// The amount to lower the minInherentFitness score by
	// eg. to promote diversity in fitness, despite worse value
	// Note, only used if "normalize fitness to population" is on.
	double share_margin = 0;
	

	//  0 = No Crowding
	//  1 = Deterministic Crowding (conditional)
	//  Condition: only replace if more similar to current offspring than donor.
	//  2 = Determinsitc Crowding (replacement)
	//  Calculate distance between previous and donor. Replace solution that is closer.
	int crowding = 0;

	int ls_mode = 0;

	// 0 = Greedy Forward
	// 1 = Best Forward
	int solution_relinker = 0;

	string folder = "test";
	string problemName,
		   FOSName;
	string problemInstancePath = "";

	int noImprovementStretchLimit = -1;

	long long timelimitSeconds = 3600,
			  randomSeed = 42;
	
	string alphabet;
	vector<int> alphabetSize;
	int populationScheme = 2;
	int maxArchiveSize = 1000000;
	int maximumNumberOfGOMEAs  = 100,
		IMSsubgenerationFactor = 4,
	    basePopulationSize     = 2,
	    populationSize = 1, 
	    maxGenerations = 200,
	    maxGenerationsWithoutImprovement = 10;
	int hillClimber = 0,
	    donorSearch = 0,
	    reuseOffsprings = 0,
	    tournamentSelection = 0;
    long long maxEvaluations = 1e+12;
    string functionName="";
	mt19937 rng;

	int useSurrogate=0;
	int maxEvalsWithoutElitistUpdate = 10;
	double delta = 0.02;

	bool parseCommandLine(int argc, char **argv);
	void checkOptions();
	void printUsage();
	void printOverview();
	void writeOverviewToFile();
	void printOverview(ostream &s);
};
