//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
//
// This research code was modified as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
//
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt (modifications for relevant article), Arkadiy Dushatskiy (original developer)

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