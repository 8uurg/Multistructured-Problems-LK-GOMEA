#pragma once
using namespace std;

#include "Config.hpp"
#include "Individual.hpp"
#include "shared.hpp"

class BitflipLS {
public:
    BitflipLS(Config *config_, sharedInformation *info_, Problem *problem_);
    ~BitflipLS();
    void optimize(Individual *individual_, size_t populationSize);
    // void shuffle_order(mt19937 *rng);
    void evaluateSolution(Individual *new_solution, Individual *orig_solution, vector<int> &changedGenes, size_t populationSize);
    void checkVTR(Individual *solution, size_t populationSize);
    void checkTimeLimit();

    Config *config;
    sharedInformation *info;
    Problem *problem;
    vector<size_t> sequence;
};