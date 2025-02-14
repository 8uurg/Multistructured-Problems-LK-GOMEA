//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
//
// This research code was modified as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
//
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt (modifications for relevant article), Arkadiy Dushatskiy (original developer)

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