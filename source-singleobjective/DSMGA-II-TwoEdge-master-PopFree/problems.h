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
#include <iostream>
using namespace std;

class Problem
{
public:
	int numberOfVariables;
	bool print_problem_name_to_stdout = false;
	
	Problem(){};
	virtual ~Problem(){};
	virtual void initializeProblem(int numberOfVariables)=0;
	virtual double calculateFitness(unsigned long *solution)=0;
 
	inline int quotientLong(int a) {
	    return (a / (sizeof(unsigned long) * 8) );
	}

	inline int remainderLong(int a) {
	    return (a & (sizeof(unsigned long) * 8 - 1));
	}

    int getVal (unsigned long *gene, int index)
    {
	    int q = quotientLong(index);
	    int r = remainderLong(index);

	    if ( (gene[q] & (1lu << r)) == 0 )
	        return 0;
	    else
	        return 1;
	}


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
		{
			if (print_problem_name_to_stdout)
				cout<<"creating concatenated Deceptive Trap with trap size=" << k << " and shift=" << s << endl;
		}
		else
		{
			if (k != 10 && k != 6)
			{
				cout << "Bimodal trap with k=" << k << " not implemented!" << endl;
				exit(0);
			}
			if (print_problem_name_to_stdout)
				cout<<"creating bimodal concatenated Deceptive Trap with trap size=" << k << " and shift=" << s << endl;
		}
	}
	void initializeProblem(int numberOfVariables_);
	double calculateFitness(unsigned long *solution);
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
		if (print_problem_name_to_stdout)
			cout<<"creating ADF\n";
	}

	void initializeProblem(int numberOfVariables_);
	double calculateFitness(unsigned long *solution);
};

class hierarchialIfAndOnlyIf:public Problem
{
public:
	hierarchialIfAndOnlyIf()
	{
		if (print_problem_name_to_stdout)
			cout<<"creating hierarchialIfAndOnlyIf\n";
	}
	void initializeProblem(int numberOfVariables_)
	{
		numberOfVariables = numberOfVariables_;
	};

	double calculateFitness(unsigned long *solution);
};

class maxCut:public Problem
{
	string problemInstanceFilename;
	vector<pair<pair<int, int>, double > > edges;
	vector<vector<int> > edgesForVariable;

public:
	maxCut(string problemInstanceFilename_): problemInstanceFilename(problemInstanceFilename_)
	{
		if (print_problem_name_to_stdout)
			cout<<"creating maxCut\n";
	}
	void initializeProblem(int numberOfVariables_);
	double calculateFitness(unsigned long *solution);
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
		if (print_problem_name_to_stdout)
			cout<<"creating MAXSAT " << problemInstanceFilename << endl;
	}

	void initializeProblem(int numberOfVariables_);

	double calculateFitness(unsigned long *solution);
};

class SpinGlass:public Problem
{
	string problemInstanceFilename;
	vector<vector<int> > subfunctions;
	
	vector<vector<int> > subfunctionsForVariable;

public:
	SpinGlass(string problemInstanceFilename_): problemInstanceFilename(problemInstanceFilename_)
	{
		if (print_problem_name_to_stdout)
			cout<<"creating SpinGlass " << problemInstanceFilename << endl;
	}

	void initializeProblem(int numberOfVariables_);

	double calculateFitness(unsigned long *solution);
};

double deceptiveTrap(int unitation, int k);
double bimodalDeceptiveTrap(int unitation, int k);

void createProblemInstance(int problemIndex, int numberOfVariables, Problem **problemInstance, string &instancePath, int k = 1, int s = 1);

#include "bestoftraps.h"
class BestOfTrapsProblem: public Problem
{
	BestOfTraps bot;
	std::string botInstance;

public:
	BestOfTrapsProblem(std::string botInstance): botInstance(botInstance)
	{
		if (print_problem_name_to_stdout)
			cout<<"creating BestOfTraps " << botInstance << endl;
	}

	void initializeProblem(int numberOfVariables_);

	double calculateFitness(unsigned long *solution);
};

#include "worstofmaxcut.h"
class WorstOfMaxcutProblem : public Problem
{
  private:
	std::vector<MaxCutInstance> maxcut_instances;
	size_t l;
	long evaluate_maxcut(MaxCutInstance &instance, unsigned long *solution);

  public:
  	WorstOfMaxcutProblem(std::string instance_str);

	void initializeProblem(int numberOfVariables_);

	double calculateFitness(unsigned long *solution);
};

