//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
//
// This research code was modified as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
//
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt (modifications for relevant article), Arkadiy Dushatskiy (original developer)

#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <deque>
#include <random>
using namespace std;

#include "Individual.hpp"
#include "utils.hpp"
#include "problems.hpp"

class FOS
{	
public:
	vector<vector<int> > FOSStructure;	
	size_t numberOfVariables;
	vector<int> alphabetSize;
	vector<vector<int> > graph;
	
	vector<int> improvementCounters;
	vector<int> usageCounters;

	FOS(size_t numberOfVariables_, vector<int> &alphabetSize_): numberOfVariables(numberOfVariables_), alphabetSize(alphabetSize_)
	{}

	virtual ~FOS(){};

	size_t FOSSize()
	{
		return FOSStructure.size();
	}

	size_t FOSElementSize(int i)
	{
		return FOSStructure[i].size();
	}
	
	virtual void learnFOS(vector<Individual*> &population, vector<vector<int> > *VIG = NULL, mt19937 *rng = NULL) = 0;
	void writeToFileFOS(string folder, int populationIndex, int generation);
	void writeFOSStatistics(string folder, int populationIndex, int generation);
	void setCountersToZero();
	void shuffleFOS(vector<int> &indices, mt19937 *rng);
	void sortFOSAscendingOrder(vector<int> &indices);
	void sortFOSDescendingOrder(vector<int> &indices);
	void orderFOS(int orderingType, vector<int> &indices, mt19937 *rng);

	virtual void buildGraph(double thresholdValue, mt19937 *rng){};
	virtual void buildGraphGlobal(double thresholdValue){};

	virtual void writeMIMatrixToFile(string folder, int populationIndex, int generation){};

	
	/* BEGIN: CHANGES FOR MEMORY USAGE REDUCTION */
	virtual void shrinkMemoryUsage(){};
	/* END: CHANGES FOR MEMORY USAGE REDUCTION */
};

class ProblemFOS: public FOS
{
public:
	ProblemFOS(size_t numberOfVariables_, vector<int> &alphabetSize_, Problem *problem);
	// FOS is initialized once, using the current problem.
	void learnFOS(vector<Individual*> &population, vector<vector<int> > *VIG = NULL, mt19937 *rng = NULL) {};
};


class LTFOS: public FOS
{
private:
	vector<vector<double> > MI_Matrix;
	vector<vector<double> > S_Matrix;
	bool filtered;
	int similarityMeasure;
	int determineNearestNeighbour(int index, vector< vector< int > > &mpm);
	void computeMIMatrix(vector<Individual*> &population);
	void computeNMIMatrix(vector<Individual*> &population);
	void computeRandomMatrix(mt19937 *rng);
	void estimateParametersForSingleBinaryMarginal(vector<Individual*> &population, vector<size_t> &indices, size_t  &factorSize, vector<double> &result);
	void prepareMatrices();

public:	
	LTFOS(size_t numberOfVariables_, vector<int> &alphabetSize_, int similarityMeasure, bool filtered=false);
	~LTFOS(){};

	void learnFOS(vector<Individual*> &population, vector<vector<int> > *VIG = NULL, mt19937 *rng = NULL);
	void buildGraph(double thresholdValue, mt19937 *rng);
	void buildGraphGlobal(double thresholdValue);
	void buildGraphStatTest(solutionsArchive *solutions, double p_value_threshold);
	void writeMIMatrixToFile(string folder, int populationIndex, int generation);
	void shrinkMemoryUsage();
};

bool FOSNameByIndex(size_t FOSIndex, string &FOSName);
void createFOSInstance(size_t FOSIndex, FOS **FOSInstance, size_t numberOfVariables, vector<int> &alphabetSize, int similarityMeasure, Problem *problem);
