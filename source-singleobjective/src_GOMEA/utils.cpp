//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
//
// This research code was modified as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
//
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt (modifications for relevant article), Arkadiy Dushatskiy (original developer)

#include "utils.hpp"

void prepareFolder(string &folder)
{
	cout << "preparing folder: " << folder << endl;
	if (filesystem::exists(folder))
	{
		filesystem::remove_all(folder);
	}
	filesystem::create_directories(folder);
	filesystem::create_directories(folder + "/fos");
	filesystem::create_directories(folder + "/populations");
}

void initElitistFile(string &folder, int populationScheme, int populationSize)
{
	ofstream outFile(folder + "/elitists.txt", ofstream::out);
	if (outFile.fail())
	{
		cerr << "Problems with opening file " << folder + "/elitists.txt!\n";
		exit(0);
	}
	// if (populationScheme == 3)
	// outFile << "population size: " << populationSize << endl;

	outFile << "#Evaluations " << "Time,millisec. " << "Fitness " << "IsKey " << "PopulationSize " << "Solution" << endl;
	outFile.close();
}
void writeElitistSolutionToFile(string &folder, long long numberOfEvaluations, long long time, Individual *solution)
{
	writeElitistSolutionToFile(folder, numberOfEvaluations, time, solution, false, 1);
}

void writeElitistSolutionToFile(string &folder, long long numberOfEvaluations, long long time, Individual *solution, bool is_key, size_t populationSize)
{
	ofstream outFile(folder + "/elitists.txt", ofstream::app);
	if (outFile.fail())
	{
		cerr << "Problems with opening file " << folder + "/elitists.txt!\n";
		exit(0);
	}

	outFile << (int)numberOfEvaluations << " " << time << " " <<  fixed << setprecision(6) << solution->fitness << " " << is_key << " " << populationSize << " ";
	for (size_t i = 0; i < solution->genotype.size(); ++i)
		outFile << +solution->genotype[i];
	outFile << endl;

	outFile.close();
}

void solutionsArchive::checkAlreadyEvaluated(vector<char> &genotype, archiveRecord *result)
{
	result->isFound = false;

	unordered_map<vector<char>, double, hashVector >::iterator it = archive.find(genotype);
	if (it != archive.end())
	{
		result->isFound = true;
		result->value = it->second;
 	}
}

void solutionsArchive::insertSolution(vector<char> &genotype, double fitness)
{
	// #if DEBUG
	// 	cout << "Inserting solution ";
	// 	for (size_t i = 0; i < solution.size(); ++i)
	// 		cout << solution[i];
	// #endif
	if (archive.size() >= maxArchiveSize)
		return;
	archive.insert(pair<vector<char>, double> (genotype, fitness));
}

bool isPowerOfK(int n, int k)
{
	double logNBaseK = log(n) / log(k);
	return (ceil(logNBaseK) == floor(logNBaseK));
}

Pyramid::~Pyramid()
{
	for (int i = 0; i < levels.size(); ++i)
	{
		for (int j = 0; j < levels[i].size(); ++j)
			delete levels[i][j];
	}
}

bool Pyramid::checkAlreadyInPyramid(vector<char> &genotype)
{
	if (find(hashset.begin(), hashset.end(), genotype) != hashset.end())
		return true;

	return false;
}

bool Pyramid::insertSolution(int level, vector<char> &genotype, double fitness)
{
	//cout << "Inserting solution " << level << endl;
	
	if (!checkAlreadyInPyramid(genotype))
	{
		if (level == levels.size())
		{
			vector<Individual*> newLevel;
			levels.push_back(newLevel);
		}
		
		Individual *newSolution = new Individual(genotype, fitness);
		levels[level].push_back(newSolution);

		hashset.insert(genotype);

		return true;
	}

	return false;
}


