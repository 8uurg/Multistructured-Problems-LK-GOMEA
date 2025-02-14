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

class PopulationGeneral
{
public:
	Config *config;
	Problem *problemInstance;
	sharedInformation *sharedInformationPointer;
	size_t populationIndex;
	size_t populationSize;

	vector<Individual*> population;
	vector<Individual*> offspringPopulation;
	vector<int> noImprovementStretches;

	FOS *populationFOS;
	bool terminated;
	double averageFitness;
	size_t numberOfGenerations;
	FOS *FOSInstance = NULL;
	vector<vector<double> > matrix;

	PopulationGeneral(Config *config_, Problem *problemInstance_, sharedInformation *sharedInformationPointer_, size_t populationIndex_, size_t populationSize_);
	virtual ~PopulationGeneral(){};

	void tournamentSelection(int k, vector<Individual*> &population, vector<Individual*> &offspringPopulation);
	void hillClimberSingle(Individual *solution);	
	void hillClimberMultiple(Individual *solution);

	void calculateAverageFitness();	
	void copyOffspringToPopulation();
	void evaluateSolution(Individual *solution);
	bool GOM(size_t offspringIndex, Individual *backup);
	void findNeighbors(vector<vector< int> > &neighbors);
	bool conditionalGOM(size_t offspringIndex, Individual *backup, vector<vector<int> > &neighbors);	
	bool FI(size_t offspringIndex, Individual *backup);
	bool conditionalFI(size_t offspringIndex, Individual *backup, vector<vector<int> > &neighbors);
	void updateElitistAndCheckVTR(Individual *solution);
	void checkTimeLimit();
	int compareSolutions(Individual *x, Individual*y);
};

