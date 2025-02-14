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
#include <string>
#include <fstream>
#include <deque>
#include <algorithm>
#include <unordered_set>
#include <set>
#include <cassert>
#include <filesystem>

#include <Python.h>

using namespace std;

#include "Individual.hpp"
#include "utils.hpp"

class Config;
#include "Config.hpp"

class Problem
{
public:
	int numberOfVariables;
	int usePartialEvaluations;
	vector<vector<int> > graph;
	
	Problem(){};
	virtual ~Problem(){};
	virtual void initializeProblem(Config *config, int numberOfVariables)=0;
	virtual double calculateFitness(Individual *solution)=0;
	virtual int getEvals(){return -1;};
	virtual double calculateFitnessPartialEvaluations(Individual *solution, Individual *solutionBefore, vector<int> &touchedGenes, double fitnessBefore);
	virtual bool isKeySolution(vector<char> &genotype, double fitness)
	{
		static bool warnOnce = true;
		if (warnOnce)
		{
			cout << "[WARNING] Chosen problem does NOT provide any key solutions." << endl;
			warnOnce = false;
		}
		return false;
	};
	virtual vector<vector<int>> getProblemFOS()
	{
		cerr << "Problem used does not define a problem specific FOS.\n";
		cerr << "Please define within the problem class, or avoid using this mode." << endl;
		exit(0);
	}
};

class oneMax:public Problem
{
public:
	oneMax(){cout<<"creating oneMax\n";}
	void initializeProblem(Config *config, int numberOfVariables_)
	{
		numberOfVariables = numberOfVariables_;
	};
	double calculateFitness(Individual *solution);
	double calculateFitnessPartialEvaluations(Individual *solution, Individual *solutionBefore, vector<int> &touchedGenes, double fitnessBefore);
};

// TODO: Rename this. This is the name of another function
// (which alternates 010101 for the optimum)
class zeroOneMax:public Problem
{
public:
	zeroOneMax(){cout<<"creating zeroOneMax\n";}
	void initializeProblem(Config *config, int numberOfVariables_)
	{
		numberOfVariables = numberOfVariables_;
	};
	double calculateFitness(Individual *solution);
	double calculateFitnessPartialEvaluations(Individual *solution, Individual *solutionBefore, vector<int> &touchedGenes, double fitnessBefore);
};

class closestDistanceToGridPoints:public Problem
{
private:
	int k;
public:
	closestDistanceToGridPoints(int k_): k(k_)
	{
		cout<<"creating closestDistanceToGridPoints" << endl;
	}
	void initializeProblem(Config *config, int numberOfVariables_)
	{
		numberOfVariables = numberOfVariables_;
	};
	double calculateFitness(Individual *solution);
	double calculateFitnessPartialEvaluations(Individual *solution, Individual *solutionBefore, vector<int> &touchedGenes, double fitnessBefore);
};

class bimodalModifiedTrap: public Problem
{
	int k, s;
	// This bimodal
	vector<vector<int>> trapsForVariable_a; //determines traps which each variables belongs to
	vector<vector<int>> trapsForVariable_b;
public:
	bimodalModifiedTrap(int k_, int s_): k(k_), s(s_)
	{
		cout<<"creating bimodalModifiedTrap" << endl;
	}
	void initializeProblem(Config *config, int numberOfVariables_);
	double calculateFitness(Individual *solution);
	double calculateFitnessPartialEvaluations(Individual *solution, Individual *solutionBefore, vector<int> &touchedGenes, double fitnessBefore);
	bool isKeySolution(vector<char> &genotype, double fitness);
};

class maxConcatenatedPermutedTraps: public Problem
{
	// Block size
	int k;
	// Seed (for rng)
	int s;
	// Number of functions.
	int fn;

	vector<vector<size_t>> permutations;
	vector<vector<char>> optima;

public:
	maxConcatenatedPermutedTraps(int k_, int s_, int fn_): k(k_), s(s_), fn(fn_)
	{
		cout<<"creating maxConcatenatedPermutedTraps" << endl;
	}
	void initializeProblem(Config *config, int numberOfVariables_);
	double calculateFitness(Individual *solution);
	double calculateFitnessPartialEvaluations(Individual *solution, Individual *solutionBefore, vector<int> &touchedGenes, double fitnessBefore);
	bool isKeySolution(vector<char> &genotype, double fitness);
	vector<vector<int>> getProblemFOS();

private:
	vector<double> calculateFitnessSubfunctions(vector<char> &genotype);
};

class concatenatedDeceptiveTrap:public Problem
{
	int k, s;
	bool bimodal;
	vector<vector<int> > trapsForVariable; //determines traps which each variables belongs to
public:
	concatenatedDeceptiveTrap(int k_, int s_, bool bimodal_): k(k_), s(s_), bimodal(bimodal_)
	{
		if (not bimodal_)
			cout<<"creating concatenated Deceptive Trap with trap size=" << k << " and shift=" << s << endl;
		else
		{
			if (k != 10 && k != 6)
			{
				cout << "Bimodal trap with k=" << k << " not implemented!" << endl;
				exit(0);
			}
			cout<<"creating bimodal concatenated Deceptive Trap with trap size=" << k << " and shift=" << s << endl;
		}
	}
	void initializeProblem(Config *config, int numberOfVariables_);
	double calculateFitness(Individual *solution);
	double calculateFitnessPartialEvaluations(Individual *solution, Individual *solutionBefore, vector<int> &touchedGenes, double fitnessBefore);
};

struct NKSubfunction
{
	vector<int> variablesPositions;
	vector<double> valuesTable;
};

class ADF:public Problem
{
	string problemInstanceFilename;
	vector<NKSubfunction> subfunctions;
	vector<vector<int> > subfunctionsForVariable;

public:
	ADF(string problemInstanceFilename_): problemInstanceFilename(problemInstanceFilename_)
	{
		cout<<"creating ADF\n";
	}

	void initializeProblem(Config *config, int numberOfVariables_);

	double calculateFitness(Individual *solution);
	double calculateFitnessPartialEvaluations(Individual *solution, Individual *solutionBefore, vector<int> &touchedGenes, double fitnessBefore);
	vector<vector<int>> getProblemFOS();
};

class hierarchialDeceptiveTrap:public Problem
{
	int k;
	vector<int> transform;
public:
	hierarchialDeceptiveTrap(int k_): k(k_)
	{
		cout<<"creating hierarchialDeceptiveTrap\n";
	}

	void initializeProblem(Config *config, int numberOfVariables_)
	{
		numberOfVariables = numberOfVariables_;
		if (!isPowerOfK(numberOfVariables, k))
		{
			cerr << "Number of bits should be a power of k! " << numberOfVariables << " is not a power of " << k << endl;
			exit(0);
		}
		transform.resize(numberOfVariables);		
	};

	double generalTrap(int unitation, double leftPeak, double rightPeak);
	double calculateFitness(Individual *solution);
	//double calculateFitnessPartialEvaluations(Individual *solution, Individual *solutionBefore, vector<double> &touchedGenes, double fitnessBefore);
};

class hierarchialIfAndOnlyIf:public Problem
{
public:
	hierarchialIfAndOnlyIf()
	{
		cout<<"creating hierarchialIfAndOnlyIf\n";
	}
	void initializeProblem(Config *config, int numberOfVariables_)
	{
		numberOfVariables = numberOfVariables_;
		if (!isPowerOfK(numberOfVariables, 2))
		{
			cerr << "Number of bits should be a power of 2! " << numberOfVariables<< " is not a power of 2" << endl;
			exit(0);
		}

		graph.resize(numberOfVariables);
		for (int i = 0; i < numberOfVariables; ++i)
		{
			for (int j = 0; j < numberOfVariables; ++j)
			{
				if (i == j)
					continue;
				graph[i].push_back(j);
			}
		}

	};

	double calculateFitness(Individual *solution);
};

class maxCut:public Problem
{
	string problemInstanceFilename;
	vector<pair<pair<int, int>, double > > edges;
	vector<vector<int> > edgesForVariable;

public:
	maxCut(string problemInstanceFilename_): problemInstanceFilename(problemInstanceFilename_)
	{
		cout<<"creating maxCut\n";
	}
	void initializeProblem(Config *config, int numberOfVariables_);
	double calculateFitness(Individual *solution);
	double calculateFitnessPartialEvaluations(Individual *solution, Individual *solutionBefore, vector<int> &touchedGenes, double fitnessBefore);
};


class Clustering:public Problem
{
	string problemInstanceFilename;
	vector<vector<double> > points;
	vector<vector<double> > distances;
	
	int Dim;
public:
	Clustering(string problemInstanceFilename_): problemInstanceFilename(problemInstanceFilename_)
	{
		cout<<"creating Clustering\n";
	}
	void initializeProblem(Config *config, int numberOfVariables_);
	double calculateFitness(Individual *solution);
};

class MAXSAT:public Problem
{
	string problemInstanceFilename;
	vector<vector<int> > subfunctions;
	vector<vector<int> > signs;
	
	vector<vector<int> > subfunctionsForVariable;

public:
	MAXSAT(string problemInstanceFilename_): problemInstanceFilename(problemInstanceFilename_)
	{
		cout<<"creating MAXSAT\n";
	}

	void initializeProblem(Config *config, int numberOfVariables_);

	double calculateFitness(Individual *solution);

	vector<vector<int>> getProblemFOS();
};

class SpinGlass:public Problem
{
	string problemInstanceFilename;
	vector<vector<int> > subfunctions;
	
	vector<vector<int> > subfunctionsForVariable;

public:
	SpinGlass(string problemInstanceFilename_): problemInstanceFilename(problemInstanceFilename_)
	{
		cout<<"creating SpinGlass\n";
	}

	void initializeProblem(Config *config, int numberOfVariables_);

	double calculateFitness(Individual *solution);
};

// BEGIN Best-of-Traps

// Definition
struct PermutedRandomTrap
{
    int number_of_parameters;
    int block_size;
    std::vector<size_t> permutation;
    std::vector<char> optimum;
};

struct BestOfTraps
{
    std::vector<PermutedRandomTrap> permutedRandomTraps;
};

class BestOfTrapsProblem:public Problem
{
private:
	std::string instance;
	BestOfTraps bot;

public:
	BestOfTrapsProblem(std::string _instance): instance(_instance)
	{
		cout<<"creating BestOfTraps" << endl;
	}
	void initializeProblem(Config *config, int numberOfVariables_);
	double calculateFitness(Individual *solution);
	double calculateFitnessPartialEvaluations(Individual *solution, Individual *solutionBefore, vector<int> &touchedGenes, double fitnessBefore);
};

class WorstOfTrapsProblem:public Problem
{
private:
	std::string instance;
	BestOfTraps bot;

public:
	WorstOfTrapsProblem(std::string _instance): instance(_instance)
	{
		cout<<"creating BestOfTraps" << endl;
	}
	void initializeProblem(Config *config, int numberOfVariables_);
	double calculateFitness(Individual *solution);
	double calculateFitnessPartialEvaluations(Individual *solution, Individual *solutionBefore, vector<int> &touchedGenes, double fitnessBefore);
};

struct MaxCutInstance
{
	size_t l;
	std::vector<std::tuple<size_t, size_t, long>> edges;
};

class WorstOfMaxcutProblem : public Problem
{
  private:
	std::vector<MaxCutInstance> maxcut_instances;
	size_t l;

  public:
  	WorstOfMaxcutProblem(std::string instance_str);

        void initializeProblem(Config * /* config */, int numberOfVariables_);

        double calculateFitness(Individual *solution);
        double calculateFitnessPartialEvaluations(Individual *solution,
                                                  Individual *solutionBefore,
                                                  vector<int> &touchedGenes,
                                                  double fitnessBefore);
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////


class PythonFunction:public Problem
{
	string functionName, instancePath;
	PyObject *module, *functionClass, *functionInstance, *fitnessFunction, *getEvalsFunction;

public:
	PythonFunction(string functionName_, string instancePath_): functionName(functionName_), instancePath(instancePath_)
	{
		cout<<"creating Python Function " << functionName << endl;
	}
	~PythonFunction()
	{
		Py_DECREF(module);
		Py_DECREF(functionClass);
		Py_DECREF(functionInstance);
	}

	void initializeProblem(Config *config, int numberOfVariables_);
	double calculateFitness(Individual *solution);
	int getEvals();
};


double deceptiveTrap(int unitation, int k);
double bimodalDeceptiveTrap(int unitation, int k);

void createProblemInstance(int problemIndex, int numberOfVariables, Config *config, Problem **problemInstance, string &instancePath, int k = 1, int s = 1);
bool problemNameByIndex(Config *config, string &problemName);

