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

#include "utils.hpp"
#include "time.hpp"

struct sharedInformation
{
	double numberOfEvaluations, numberOfSurrogateEvaluations;
	long long startTimeMilliseconds;
	double elitistSolutionHittingTimeMilliseconds,
	       elitistSolutionHittingTimeEvaluations;

	Individual elitist;
	double inherentElitistFitness;
	solutionsArchive *evaluatedSolutions;
	solutionsArchive *vtrUniqueSolutions;
	bool firstEvaluationEver;
	double percentileThreshold;
	Pyramid *pyramid;
	
	sharedInformation(int maxArchiveSize)
	{
		numberOfEvaluations = 0;
		startTimeMilliseconds = getCurrentTimeStampInMilliSeconds();
		firstEvaluationEver = true;
		inherentElitistFitness = INFINITY;
		evaluatedSolutions = new solutionsArchive(maxArchiveSize);
		vtrUniqueSolutions = new solutionsArchive(INT_MAX);
		pyramid = new Pyramid();
		//all_graphs.resize(1);
	}

	~sharedInformation()
	{
		delete evaluatedSolutions;
		delete vtrUniqueSolutions;
		delete pyramid;
	}
};