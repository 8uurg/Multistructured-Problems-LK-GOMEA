//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
//
// This research code was modified as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
//
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt (modifications for relevant article), Arkadiy Dushatskiy (original developer)

#pragma once

#include <string> 
#include <iostream>
using namespace std;

#include "Config.hpp"
#include "Individual.hpp"
#include "shared.hpp"
#include "problems.hpp"

class GOMEA
{
public:
	Config *config;
	Problem *problemInstance = NULL;
	sharedInformation *sharedInformationInstance = NULL;
	
	GOMEA(Config *config_): config(config_){};
	virtual ~GOMEA(){};

	virtual void run() = 0;
	double readVTR();
};