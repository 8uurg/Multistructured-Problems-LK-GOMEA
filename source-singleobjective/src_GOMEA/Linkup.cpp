//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
//
// This research code was modified as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
//
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt (modifications for relevant article), Arkadiy Dushatskiy (original developer)

#include "Linkup.hpp";

Linkup::Linkup(Config *config_, sharedInformation *info_, Problem *problem_) : config(config_), info(info_), problem(problem_)
{
}

Linkup::~Linkup()
{
}

bool Linkup::relink(Individual *from, Individual *to)
{
    switch (config->solution_relinker)
    {
    case 0:
        return Linkup::relink_forward_greedy(from, to);
        break;
    case 1:
        return Linkup::relink_forward_best(from, to);
        break;
    default:
        throw customException("relinker unknown");
        break;
    }
}

bool Linkup::relink_forward_greedy(Individual *from, Individual *to)
{
    Individual current;
    Individual next;
    current = *from;
    next = *from;

    vector<int> changedGenes = {0};
    vector<size_t> indices;
    indices.resize(from->alphabetSize.size());
    iota(indices.begin(), indices.end(), 0);

    while (indices.size() > 0)
    {
        size_t x = 0;
        bool success = false;
        for (size_t p = 0; p < indices.size(); ++p)
        {
            int index = indices[p];
            // Solutions current and next should still be identical at this point.
            assert(current.genotype[index] == next.genotype[index]);

            // Skip over items with identical values.
            if (current.genotype[index] == to->genotype[index])
                continue;

            // Try changing the bit.
            next.genotype[index] = to->genotype[index];
            changedGenes[0] = index;
            // Evaluate the change
            evaluateSolution(&next, &current, changedGenes);

            bool isHill = next.fitness < from->fitness && next.fitness < to->fitness;

            if (!isHill)
            {
                // Step has been performed.
                success = true;
                current = next;
            }
            else
            {
                // Undo change
                next = current;
                // Rewite index to indices (to compact the list of remaining bitflips)
                indices[x] = index;
                // Progress x
                ++x;
            }
        }
        indices.resize(x);
        if (!success && indices.size() != 0)
        {
            return false;
        }
    }

    return true;
}

bool Linkup::relink_forward_best(Individual *from, Individual *to)
{
    Individual current;
    Individual next;
    Individual next_best;

    current = *from;
    next_best = *from;
    next = *from;

    size_t steps = 0;

    vector<int> changedGenes = {0};
    vector<size_t> indices;
    indices.resize(from->alphabetSize.size());
    iota(indices.begin(), indices.end(), 0);

    while (indices.size() > 0)
    {
        size_t x = 0;
        bool success = false;

        for (size_t p = 0; p < indices.size(); ++p)
        {
            int index = indices[p];
            // Solutions current and next should still be identical at this point.
            assert(current.genotype[index] == next.genotype[index]);

            // Skip over items with identical values.
            if (current.genotype[index] == to->genotype[index])
                continue;

            // Try changing the bit.
            next.genotype[index] = to->genotype[index];
            changedGenes[0] = index;
            // Evaluate the change
            evaluateSolution(&next, &current, changedGenes);

            bool isHill = next.fitness < from->fitness && next.fitness < to->fitness;

            if (!isHill && (!success || next_best.fitness < next.fitness))
            {
                // Not a hill & closer, therefore a potential next step.
                // We want to find the best solution in the (restricted) neighborhood,
                // even taking a worsening step if neccesary to get closer.
                // If success is still false, we haven't done any step yet, so update the next best
                // as well.

                success = true;
                next_best = next;
            }
            // Unlike in the greedy case, we do not know if the change will actually stay.
            // So in any case preserve the index.
            // The one chosen will be removed by continuing over it the next time around.
            // Rewite index to indices (to compact the list of remaining bitflips)
            indices[x] = index;
            // Progress x
            ++x;
            // Undo change
            next = current;
        }
        indices.resize(x);

        steps += 1; 

        // We were unable to perform a change, current is surrounded by hills.
        if (!success && indices.size() != 0)
        {
            return false;
        }
        // Update current solution.
        current = next_best;
        next = next_best;
    }

    return true;
}

/// Copied and modified from PopulationGeneral.cpp

void Linkup::evaluateSolution(Individual *new_solution, Individual *orig_solution, vector<int> &changedGenes)
{
    checkTimeLimit();

    archiveRecord searchResult;

    if (config->saveEvaluations)
        info->evaluatedSolutions->checkAlreadyEvaluated(new_solution->genotype, &searchResult);

    if (searchResult.isFound)
    {
        new_solution->fitness = searchResult.value;
    }
    else
    {
        if (orig_solution == NULL)
        {
            problem->calculateFitness(new_solution);
        }
        else
        {
            problem->calculateFitnessPartialEvaluations(new_solution, orig_solution, changedGenes, orig_solution->fitness);
        }

        if (config->problemIndex == 10)
            info->numberOfEvaluations = problem->getEvals();
        else
            info->numberOfEvaluations++;

        if (config->saveEvaluations)
            info->evaluatedSolutions->insertSolution(new_solution->genotype, new_solution->fitness);

        checkVTR(new_solution);

        if (info->numberOfEvaluations >= config->maxEvaluations)
        {
            cout << "Max evals limit reached! Terminating...\n";
            throw customException("max evals");
        }
    }
}

void Linkup::checkTimeLimit()
{
    if (getMilliSecondsRunningSinceTimeStamp(info->startTimeMilliseconds) > config->timelimitSeconds * 1000)
    {
        cout << "TIME LIMIT REACHED!" << endl;
        throw customException("time");
    }
}

void Linkup::checkVTR(Individual *solution)
{
    bool improved = info->firstEvaluationEver || (solution->fitness > info->elitist.fitness);
    bool vtr_hit = (solution->fitness >= config->vtr);
    bool unique_vtr_hit = false;
    if (improved || vtr_hit)
    {
        info->elitistSolutionHittingTimeMilliseconds = getMilliSecondsRunningSinceTimeStamp(info->startTimeMilliseconds);
        info->elitistSolutionHittingTimeEvaluations = info->numberOfEvaluations;

        info->elitist = *solution;

        /* Check the VTR */
        if (solution->fitness >= config->vtr)
        {
            archiveRecord rec;
            info->vtrUniqueSolutions->checkAlreadyEvaluated(solution->genotype, &rec);
            if (!rec.isFound)
            {
                info->vtrUniqueSolutions->insertSolution(solution->genotype, solution->fitness);

                // Stop once all unique occurences have been hit.
                if (info->vtrUniqueSolutions->archive.size() >= config->vtr_n_unique)
                {
                    cout << "LAST UNIQUE VTR HIT!" << endl;
                    writeElitistSolutionToFile(config->folder, info->elitistSolutionHittingTimeEvaluations, info->elitistSolutionHittingTimeMilliseconds, solution);
                    throw customException("vtr");
                }
                else
                {
                    unique_vtr_hit = true;
                    cout << "UNIQUE VTR HIT!" << endl;
                }
            }
        }
        if (improved || unique_vtr_hit)
            writeElitistSolutionToFile(config->folder, info->elitistSolutionHittingTimeEvaluations, info->elitistSolutionHittingTimeMilliseconds, solution);
    }

    info->firstEvaluationEver = false;
}