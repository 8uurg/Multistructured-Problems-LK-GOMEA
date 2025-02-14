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
#include "Population.hpp"
#include "problems.hpp"
#include "shared.hpp"
#include "gomea.hpp"

class gomeaIMS: public GOMEA
{
public:
	int maximumNumberOfGOMEAs;
	int generationsWithoutImprovement;
	int IMSsubgenerationFactor, basePopulationSize, numberOfGOMEAs, numberOfGenerationsIMS, minimumGOMEAIndex;

	vector<Population*> GOMEAs;

	gomeaIMS(Config *config_);
	~gomeaIMS();
	
	void initializeNewGOMEA();
	bool checkTermination();
	void generationalStepAllGOMEAs();
	bool checkTerminationGOMEA(int GOMEAIndex);
	void GOMEAGenerationalStepAllGOMEAsRecursiveFold(int indexSmallest, int indexBiggest);
	void run();
};