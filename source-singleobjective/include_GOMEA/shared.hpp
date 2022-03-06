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