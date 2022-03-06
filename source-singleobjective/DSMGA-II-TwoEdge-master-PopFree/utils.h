#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>
using namespace std;

int verbosity = 0;

void prepareFolder(string &folder)
{
    if (verbosity >= 1)
        cout << "preparing folder: " << folder << endl;
    if (filesystem::exists(folder))
    {
        filesystem::remove_all(folder);
    }
    filesystem::create_directories(folder);
    filesystem::create_directories(folder + "/fos");
}

void initElitistFile(string &folder, int /* populationSize */)
{
    ofstream outFile(folder + "/elitists.txt", ofstream::out);
    if (outFile.fail())
    {
        cerr << "Problems with opening file " << folder + "/elitists.txt!\n";
        exit(0);
    }
    // outFile << "population size: " << populationSize << endl;

    outFile << "#Evaluations " << "Time,millisec. " << "Fitness " << "IsKey " << "PopulationSize " << "Solution" << endl;
    outFile.close();
}




