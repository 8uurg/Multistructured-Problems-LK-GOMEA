//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
//
// This research code was modified as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
//
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt (modifications for relevant article), Arkadiy Dushatskiy (original developer)

#pragma once

#include <iostream> 
#include <vector>
#include <random>
#include <cassert>
using namespace std;

class Individual
{
public:
	size_t numberOfVariables;
	size_t numberOfBehavioralDims;
	vector<int> alphabetSize;
	vector<char> genotype;
	double fitness;

	// Fitness of the underlying solution.
	double inherentFitness;
	// Novelty of the behavior
	double inherentNovelty;

	vector<int> behavior;

	Individual() {};

	Individual(size_t numberOfVariables_, vector<int> &alphabetSize_): numberOfVariables(numberOfVariables_), alphabetSize(alphabetSize_)
	{
		genotype.resize(numberOfVariables_);
		fill(genotype.begin(), genotype.end(), 2);
		behavior.resize(numberOfVariables_);
		fill(behavior.begin(), behavior.end(), 4);
	}

	Individual(vector<char> &genotype_, double fitness_): fitness(fitness_)
	{
		numberOfVariables = genotype_.size();
		genotype.resize(numberOfVariables);
		copy(genotype_.begin(), genotype_.end(), genotype.begin());
		behavior.resize(numberOfVariables);
		fill(behavior.begin(), behavior.end(), 3);
	}

	Individual(vector<char> &genotype_, double fitness_, vector<int> &behavior_): fitness(fitness_)
	{
		numberOfVariables = genotype_.size();
		genotype.resize(numberOfVariables);
		copy(genotype_.begin(), genotype_.end(), genotype.begin());
		numberOfBehavioralDims = behavior_.size();
		behavior.resize(numberOfBehavioralDims);
		copy(behavior_.begin(), behavior_.end(), behavior.begin());
	}

	void randomInit(mt19937 *rng)
	{
		for (size_t i = 0; i < numberOfVariables; ++i)
		{
			genotype[i] = (*rng)() % alphabetSize[i];
		}
	}

	friend ostream & operator << (ostream &out, const Individual &individual);

	Individual& operator=(const Individual& other)
	{
		alphabetSize = other.alphabetSize;
		numberOfVariables = other.numberOfVariables;

		genotype = other.genotype;
		
		fitness = other.fitness;

		behavior = other.behavior;

		inherentFitness = other.inherentFitness;
		inherentNovelty = other.inherentNovelty;

		return *this;
	}

	bool operator==(const Individual& solutionB)
	{
    	for (size_t i = 0; i < numberOfVariables; ++i)
    	{
    		if (this->genotype[i] != solutionB.genotype[i])
    			return false;
    	}
    	return true;
	}
};

