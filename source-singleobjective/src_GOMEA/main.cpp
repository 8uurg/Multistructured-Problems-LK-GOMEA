#include "Config.hpp"
#include "gomea.hpp"
#include "gomeaIMS.hpp"
#include "gomeaLSNoveltyIMS.hpp"
#include "gomeaP3_MI.hpp"
#include "bitflipls.hpp"

int main(int argc, char **argv)
{
    Py_Initialize();

    Config *config = new Config();
    config->parseCommandLine(argc, argv);
    config->checkOptions();
    config->printOverview();

    config->rng.seed(config->randomSeed);


    if (config -> approach == 0 || config -> approach == 1) 
    {  
        GOMEA *gomeaInstance;
        if (config -> approach == 0)
        {
            if (config->populationScheme == 0 || config->populationScheme == 3)
                gomeaInstance = new gomeaIMS(config);
            else
                gomeaInstance = new gomeaP3_MI(config);
        }
        else
        {
            gomeaInstance = new gomeaLSNoveltyIMS(config);
        } 

        try
        {
            gomeaInstance->run();
        }
        catch (customException &ex)
        {}

        delete gomeaInstance;
    
    }
    else if (config -> approach == 2)
    {
        // 
        vector<int> test;
        Problem *problem = NULL;
        sharedInformation *info = new sharedInformation(config->maxArchiveSize);;
        prepareFolder(config->folder);
        createProblemInstance(config->problemIndex, config->numberOfVariables, config, &problem, config->problemInstancePath);

        Individual i = Individual(config->numberOfVariables, config->alphabetSize);

        BitflipLS bfls = BitflipLS(config, info, problem);

        try
        {
            bfls.optimize(&i, 1);
            cout << "Done!\n";
        }
        catch (customException &ex)
        {
            cout << "Stopped due to " << ex.what() << ".\n";
        }

        cout << "Finished @ " << i.fitness << "!\n";
        cout << "Solution: ";
        for (auto it = i.genotype.begin(); it != i.genotype.end(); ++it) 
        {
            cout << ((int) *it);
        }
        cout << ".\n";

        delete problem;
        delete info;
    } else {
        throw customException("unkApproach");
    }
    delete config;
    
    Py_Finalize();
    
    return 0;
}