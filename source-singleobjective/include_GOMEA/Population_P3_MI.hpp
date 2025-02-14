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
#include <csignal>
using namespace std;

#include "Individual.hpp"
#include "Config.hpp"
#include "shared.hpp"
#include "problems.hpp"
#include "FOS.hpp"
#include "PopulationGeneral.hpp"

class Population_P3_MI: public PopulationGeneral
{
public:
	size_t currentPyramidLevel=0;
	
	Population_P3_MI(Config *config_, Problem *problemInstance_, sharedInformation *sharedInformationPointer_, size_t GOMEAIndex_, size_t populationSize_):
		PopulationGeneral(config_, problemInstance_, sharedInformationPointer_, GOMEAIndex_, populationSize_){};	
	~Population_P3_MI();

	void makeOffspring();
	void generateOffspring();
	bool GOM(size_t offspringIndex, Individual *backup);
	bool conditionalGOM(size_t offspringIndex, Individual *backup, vector<vector<int> > &neighbors);	
};