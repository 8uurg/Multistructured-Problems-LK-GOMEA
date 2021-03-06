#pragma once
using namespace std;

#include "Config.hpp"
#include "Individual.hpp"
#include "shared.hpp"

class Linkup
{
private:
    Config *config;
    sharedInformation *info;
    Problem *problem;

    // Greedy relinking that changes a bit and takes any distance reduction,
    // as long as it is not a hill -- a solution with worse fitness than both from and to.
    // Distance is defined by hamming distance.
    // If no other choice exists than to traverse a hill, false is returned.
    // Otherwise relinking succeeds and true is returned.
    bool relink_forward_greedy(Individual *from, Individual *to);

    // Relinking that traverses the entire bitflip neighborhood and picks the solution with
    // highest fitness.
    // Distance is defined by hamming distance.
    // Progress is made as long as it is not a hill -- a solution with worse fitness than both from and to.
    // If no other choice exists than to traverse a hill, false is returned.
    // Otherwise relinking eventually arrives at to and true is returned.
    bool relink_forward_best(Individual *from, Individual *to);

public:
    Linkup(Config *config_, sharedInformation *info_, Problem *problem_);
    ~Linkup();
    bool relink(Individual *from, Individual *to);

    void evaluateSolution(Individual *new_solution, Individual *orig_solution, vector<int> &changedGenes);
    void checkVTR(Individual *solution);
    void checkTimeLimit();
};