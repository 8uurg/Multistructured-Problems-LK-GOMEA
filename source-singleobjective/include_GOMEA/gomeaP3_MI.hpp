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
#include "Population_P3_MI.hpp"
#include "problems.hpp"
#include "shared.hpp"
#include "gomea.hpp"

class gomeaP3_MI: public GOMEA
{
public:
	int maximumNumberOfGOMEAs;
	int basePopulationSize, numberOfGOMEAs;

	vector<Population_P3_MI*> GOMEAs;

	gomeaP3_MI(Config *config_);
	~gomeaP3_MI();
	
	void initializeNewGOMEA();
	bool checkTermination();
	void GOMEAGenerationalSteps(int GOMEAIndex);
	void run();
};