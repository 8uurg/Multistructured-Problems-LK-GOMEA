//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
//
// This research code was modified as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
//
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt (modifications for relevant article), Arkadiy Dushatskiy (original developer)

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
#include "PopulationNovelGeneral.hpp"

class PopulationNovelty: public PopulationNovelGeneral
{
public:

	PopulationNovelty(Config *config_, Problem *problemInstance_, sharedInformation *sharedInformationPointer_, size_t populationIndex_, size_t populationSize_):
			PopulationNovelGeneral(config_, problemInstance_, sharedInformationPointer_, populationIndex_, populationSize_){};
	
	~PopulationNovelty();

	void makeOffspring();
	void generateOffspring();
	void learnFOS(FOS *FOSInstance);
	void learnIndividualFOS(FOS *FOSInstance, size_t population_idx);
	void learnClusterFOS(FOS *FOSInstance, size_t population_idx);
	void learnIndividualBootstrapFOS(FOS *FOSInstance, size_t population_idx);
	bool GOM(size_t offspringIndex, Individual *backup);
	bool conditionalGOM(size_t offspringIndex, Individual *backup, vector<vector<int> > &neighbors);	

};