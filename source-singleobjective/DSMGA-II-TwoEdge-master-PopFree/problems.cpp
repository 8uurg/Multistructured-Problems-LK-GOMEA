#include "problems.h"

void createProblemInstance(int problemIndex, int numberOfVariables, Problem **problemInstance, string &instancePath, int k, int s)
{
	switch (problemIndex)
	{
		case 1: *problemInstance = new concatenatedDeceptiveTrap(k, s, false); break;
		case 2: *problemInstance = new ADF(instancePath); break;
		case 3: *problemInstance = new maxCut(instancePath); break;		
		case 4: *problemInstance = new hierarchialIfAndOnlyIf(); break;
		case 7: *problemInstance = new concatenatedDeceptiveTrap(k, s, true); break;
		case 8: *problemInstance = new MAXSAT(instancePath); break;
		case 9: *problemInstance = new SpinGlass(instancePath); break;
		case 10: *problemInstance = new BestOfTrapsProblem(instancePath); break;
		case 11: *problemInstance = new WorstOfMaxcutProblem(instancePath); break;

		default:
		{
			cerr << "No problem with index #" << problemIndex << " installed!\n";
			exit(0);
		};
	}
	(*problemInstance)->initializeProblem(numberOfVariables);
}


double deceptiveTrap(int unitation, int k)
{
	double result;
	if (unitation != k)
		result = k - 1 - unitation;
	else
		result = unitation;
	
	return (double)result;
}

double bimodalDeceptiveTrap(int unitation, int k)
{
	double result = 0.0;
	if (k == 10)
	{
		if (unitation == 0 || unitation == 10)
			result = 10;
		else if (unitation == 5)
			result = 9;
		else if (unitation == 1 || unitation == 9)
			result = 5;
		else if (unitation == 2 || unitation == 8)
			result = 0;
		else if (unitation == 3 || unitation == 7)
			result = 3;
		else if (unitation == 4 || unitation == 6)
			result = 6;
	}
	else if (k == 6)
	{
		if (unitation == 0 || unitation == 6)
			result = 6;
		else if (unitation == 1 || unitation == 5)
			result = 0;
		else if (unitation == 2 || unitation == 4)
			result = 2;
		else if (unitation == 3)
			result = 5;
	}	
	else
	{
		cout << "Bimodal Trap with size " << k << " not implemented!";
		exit(0);
	}

	return (double)result;
}

void concatenatedDeceptiveTrap::initializeProblem(int numberOfVariables_)
{
	numberOfVariables = numberOfVariables_;
}
	
	
double concatenatedDeceptiveTrap::calculateFitness(unsigned long *solution)
{
	double res = 0.0;
	for (int i = 0; i < numberOfVariables; i += s)
	{
		//cout << i << s << endl;
		int unitation = 0;
		for (int j = i; j < i + k; j++)
		{
			//cout << getVal(solution, j) << endl;
			unitation += getVal(solution, j%numberOfVariables);
		}

		if (!bimodal)
			res += deceptiveTrap(unitation, k);
		else if (bimodal)
			res += bimodalDeceptiveTrap(unitation, k);

		//cout << numberOfVariables << " " << i << " " << " " << k << " " << s << " " << unitation << endl;
	}

	return res;
}


// double concatenatedDeceptiveTrap::calculateFitness(unsigned long *solution)
// {
// 	double res = 0.0;
// 	for (int i = 0; i < numberOfVariables/k; i++)
// 	{
// 		//cout << i << s << endl;
// 		int unitation = 0;
// 		for (int j = i; j < numberOfVariables; j+=(numberOfVariables/k))
// 		{
// 			//cout << j << " ";
// 			//cout << getVal(solution, j) << endl;
// 			unitation += getVal(solution, j);
// 		}

// 		if (!bimodal)
// 			res += deceptiveTrap(unitation, k);
// 		else if (bimodal && k == 10)
// 			res += bimodalDeceptiveTrap10(unitation, k);

// 		//cout << numberOfVariables << " " << i << " " << " " << k << " " << s << " " << unitation << endl;
// 	}
// 	//cout << endl;

// 	return res;
// }

void ADF::initializeProblem(int numberOfVariables_)
{
	numberOfVariables = numberOfVariables_;
	subfunctionsForVariable.resize(numberOfVariables);

	//cout << problemInstanceFilename << " " << numberOfVariables << endl;

	ifstream inFile(problemInstanceFilename, ifstream::in);
	if (inFile.fail())
	{
		cout << "Problem Instance File " << problemInstanceFilename << " does not exist!\n";
		exit(0);
	}
	int N, numFunctions;
	inFile >> N >> numFunctions;
	//cout << N << " " << numFunctions << endl;

	for (int i = 0; i < numFunctions; ++i)
	{
		NKSubfunction subfunction;

		int k;
		inFile >> k;

		for (int j = 0; j < k; ++j)
		{
			int var;
			inFile >> var;
			subfunction.variablesPositions.push_back(var);
		}
		int numCombinations = 1 << k;
		subfunction.valuesTable.resize(numCombinations);
		for (int j = 0; j < numCombinations; ++j)
		{
			string combination;
			inFile >> combination;
			int index = 0, pow = 1;
			for (size_t p = combination.size()-1; p > 0; p--)
			{
				if (combination[p] == '0' || combination[p] == '1')
				{
					index += ((int)(combination[p] == '1') * pow);
					pow *= 2;
				}
			}
			inFile >> subfunction.valuesTable[index];			

		}
		subfunctions.push_back(subfunction);

		//needed for partial evaluations
		for (size_t p = 0; p < subfunction.variablesPositions.size(); ++p)
		{
			subfunctionsForVariable[subfunction.variablesPositions[p]].push_back(subfunctions.size()-1);
		}

		//VIG.push_back(subfunction.variablesPositions);

	}
	inFile.close();
	// cout << "init finished\n";
	//constructFactorization();
};

double ADF::calculateFitness(unsigned long *solution)
{
	double res = 0.0;
	
	for (size_t i = 0; i < subfunctions.size(); ++i)
	{
		int index = 0, pow = 1;
		for (int j = subfunctions[i].variablesPositions.size()-1; j >= 0; --j)
		{
			int pos = subfunctions[i].variablesPositions[j];
			index += getVal(solution, pos) * pow;
			pow *= 2;
		}
		//cout << index << " " <<  subfunctions[i].valuesTable[index] << endl;
		res += subfunctions[i].valuesTable[index];// * multiplier;
	}
	
	return (double)res;
}

double hierarchialIfAndOnlyIf::calculateFitness(unsigned long *solution)
{
	double res = 0.0;
	int blockSize = 1;
	while (blockSize <= numberOfVariables)
	{
		for (int i = 0; i < numberOfVariables; i+=blockSize)
		{
			int unitation = 0;
			for (int j = i; j < i + blockSize; ++j)
				unitation += getVal(solution, j);

			if (unitation == blockSize || unitation == 0)
				res += blockSize;
			//cout << blockSize << "  " << res << endl;
		}
		blockSize *= 2;
	}

	return res;
}

void maxCut::initializeProblem(int numberOfVariables_)
{
	numberOfVariables = numberOfVariables_;
	edgesForVariable.resize(numberOfVariables);

	ifstream inFile(problemInstanceFilename, ifstream::in);
	if (inFile.fail())
	{
		cout << "Problem Instance File " << problemInstanceFilename << " does not exist!\n";
		exit(0);
	}
	int N, numEdges;
	inFile >> N >> numEdges;

	//cout << problemInstanceFilename << " " << N << "  " << numEdges << endl;

	for (int i = 0; i < numEdges; ++i)
	{
		int v1, v2;
		double w;
		inFile >> v1 >> v2 >> w;
		edges.push_back(make_pair(make_pair(v1-1, v2-1), w));
		//cout << v1 << " " << v2 << " " << w << endl;
		edgesForVariable[v1-1].push_back(i);
		edgesForVariable[v2-1].push_back(i);		
	}

	inFile.close();
}


double maxCut::calculateFitness(unsigned long *solution)
{
	double res = 0.0;
	for (size_t i = 0; i < edges.size(); ++i)
	{
		int v1 = edges[i].first.first;
		int v2 = edges[i].first.second;
		double w = edges[i].second;

		if (getVal(solution, v1) != getVal(solution, v2))
			res += w;
	}

	return res;
}

void MAXSAT::initializeProblem(int numberOfVariables_)
{
	numberOfVariables = numberOfVariables_;
	subfunctionsForVariable.resize(numberOfVariables);

	ifstream inFile(problemInstanceFilename, ifstream::in);
	if (inFile.fail())
	{
		cout << "Problem Instance File " << problemInstanceFilename << " does not exist!\n";
		exit(0);
	}
	
	string line;
	for (int i = 0; i < 8; ++i)
		getline(inFile, line);

	while(true)
	{
		vector<int> subfunction;
		vector<int> subfunctionSigns;
		int var;
		inFile >> var;
		if (inFile.fail())
			break;
		if (var < 0)
		{
			subfunctionSigns.push_back(-1);
			var++;
		}
		else if (var > 0)
		{
			var--;
			subfunctionSigns.push_back(1);
		}
		var = abs(var);
		subfunction.push_back(var);

		while (true)
		{
			inFile >> var;
			if (var == 0)
				break;

			if (var < 0)
			{
				subfunctionSigns.push_back(-1);
				var++;
			}
			else if (var > 0)
			{
				var--;
				subfunctionSigns.push_back(1);
			}

			var = abs(var);
			subfunction.push_back(var);
		}
		for (int i = 0; i < subfunction.size(); ++i)
		{
			//cout << subfunction[i] << " | " << subfunctionSigns[i] << " ";
			subfunctionsForVariable[subfunction[i]].push_back(subfunctions.size());
		}
		//cout << endl;
		subfunctions.push_back(subfunction);
		signs.push_back(subfunctionSigns);

	}
	inFile.close();
}


double MAXSAT::calculateFitness(unsigned long *solution)
{
	long double res = 0.0;
	
	for (size_t i = 0; i < subfunctions.size(); ++i)
	{
		bool b = false;
		for (int j = 0; j < subfunctions[i].size(); ++j)
		{
			int var = subfunctions[i][j];
			int sign = signs[i][j];
			if (sign > 0 && getVal(solution, var) == 1)
				b = true;
			else if (sign < 0 && getVal(solution, var) == 0)
				b = true;			
		}
		if (b == false)
			res -= 1;		
	}
	
	return (double)res;
}



void SpinGlass::initializeProblem(int numberOfVariables_)
{
	numberOfVariables = numberOfVariables_;
	subfunctionsForVariable.resize(numberOfVariables);

	ifstream inFile(problemInstanceFilename, ifstream::in);
	if (inFile.fail())
	{
		cout << "Problem Instance File " << problemInstanceFilename << " does not exist!\n";
		exit(0);
	}
	
	double tmp1, tmp2;
	inFile >> tmp1 >> tmp2;
	
	while(true)
	{
		vector<int> subfunction;
		int var1, var2, sign;
		inFile >> var1 >> var2 >> sign;
		if (inFile.fail())
			break;
		
		var1--;
		var2--;

		subfunction.push_back(var1);
		subfunction.push_back(var2);
		subfunction.push_back(sign);		
		subfunctionsForVariable[var1].push_back(subfunctions.size());
		subfunctionsForVariable[var2].push_back(subfunctions.size());		
		subfunctions.push_back(subfunction);
		//cout << subfunction.size() << " " << var1 << " " << var2 << " " << sign << endl;
		
	}
	inFile.close();
}

double SpinGlass::calculateFitness(unsigned long *solution)
{
	long double res = 0.0;
	
	for (size_t i = 0; i < subfunctions.size(); ++i)
	{
		int cur_value = 1;
		for (int j = 0; j < 2; ++j)
		{
			if (getVal(solution, subfunctions[i][j]) == 1)
				cur_value *= 1;
			else
				cur_value *= -1;
		}
		cur_value *= subfunctions[i][2];
		res += cur_value;
	}
	
	res /= (double)numberOfVariables;
	return (double)res;
}

// Include in current compilation unit
// The lambda cannot cross compilation barriers.
#include "bestoftraps.cpp"
void BestOfTrapsProblem::initializeProblem(int numberOfVariables_)
{
	this->bot = loadBestOfTraps(botInstance, numberOfVariables_);
}

double BestOfTrapsProblem::calculateFitness(unsigned long *solution)
{
	int best_fn = 0;
	auto getValCaptured = [this, solution](int idx){return getVal(solution, idx); };
	return evaluateBestOfTraps(this->bot, getValCaptured, best_fn);
}

// Too lazy to update build!
#include "worstofmaxcut.cpp"

WorstOfMaxcutProblem::WorstOfMaxcutProblem(std::string instance_str)
{
    // Load instances
    maxcut_instances.clear();
    l = 0;
    size_t pos = 0;
    while (pos < instance_str.size()) {
        size_t next_pos = instance_str.find(';', pos);
        // if not found: use end
        if (next_pos == instance_str.npos) next_pos = instance_str.size();
        std::filesystem::path filename(instance_str.substr(pos, next_pos - pos));
        MaxCutInstance instance = load_maxcut(filename);
        l = std::max(l, instance.l);
        maxcut_instances.push_back(instance);
        pos = next_pos + 1; // + 1 to skip ';'
    }
}

void WorstOfMaxcutProblem::initializeProblem(int numberOfVariables_)
{
	assert(l == numberOfVariables_);
}

long WorstOfMaxcutProblem::evaluate_maxcut(MaxCutInstance &instance, unsigned long *solution)
{
    long v = 0;
    for (auto &[i, j, w] : instance.edges)
    {
        v += w * static_cast<long>(getVal(solution, i) != getVal(solution, j));
    }
    return v;
}

double WorstOfMaxcutProblem::calculateFitness(unsigned long *solution) {
  double f = std::numeric_limits<double>::infinity();
  for (MaxCutInstance &instance : maxcut_instances) {
    double f_m = evaluate_maxcut(instance, solution);
    if (f_m < f) {
      f = f_m;
    }
  }
  return f;
}