#pragma once

#include <vector>
#include <unordered_map>
using namespace std;

#include "Config.hpp"
#include "PopulationNovelty.hpp"
#include "problems.hpp"
#include "shared.hpp"
#include "gomea.hpp"

class gomeaLSNoveltyIMS: public GOMEA
{
public:
	int maximumNumberOfGOMEAs;
	int generationsWithoutImprovement;
	int IMSsubgenerationFactor, basePopulationSize, numberOfGOMEAs, numberOfGenerationsIMS, minimumGOMEAIndex;
	int subscheme;
	int step;
	int currentPopulationSize, currentRunCount;

	// Two options that can be enabled / disabled
	// WARNING: abort_smaller_populations_with_worse_average_fitness does lead to a performance regression
	//          due to smaller populations being aborted too early.
	bool abort_smaller_populations_with_worse_average_fitness = true;
	bool perform_cluster_aware_convergence_detection = false;

	vector<PopulationNovelty*> GOMEAs;

	gomeaLSNoveltyIMS(Config *config_);
	~gomeaLSNoveltyIMS();
	
	void initializeNewGOMEA();
	bool checkTermination();
	void generationalStepAllGOMEAs();
	void generationalStepSmallestGOMEAUntilAllTerminated(int indexSmallest, int indexBiggest);
	void generationalStepAllEvenlyGOMEAUntilAllTerminated(int indexSmallest, int indexBiggest);
	bool checkTerminationGOMEA(int GOMEAIndex);
	void GOMEAGenerationalStepAllGOMEAsRecursiveFold(int indexSmallest, int indexBiggest);
	void run();
};