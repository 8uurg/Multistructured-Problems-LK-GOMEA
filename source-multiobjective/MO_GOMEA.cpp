/**
 * MO_GOMEA.c
 *
 * IN NO EVENT WILL THE AUTHORS OF THIS SOFTWARE BE LIABLE TO YOU FOR ANY
 * DAMAGES, INCLUDING BUT NOT LIMITED TO LOST PROFITS, LOST SAVINGS, OR OTHER
 * INCIDENTIAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR THE INABILITY
 * TO USE SUCH PROGRAM, EVEN IF THE AUTHOR HAS BEEN ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGES, OR FOR ANY CLAIM BY ANY OTHER PARTY. THE AUTHOR MAKES NO
 * REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE SOFTWARE, EITHER
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT. THE
 * AUTHOR SHALL NOT BE LIABLE FOR ANY DAMAGES SUFFERED BY ANYONE AS A RESULT OF
 * USING, MODIFYING OR DISTRIBUTING THIS SOFTWARE OR ITS DERIVATIVES.
 *
 * Multi-Objective Gene-pool Optimal Mixing Evolutionary Algorithm with IMS
 *
 * In this implementation, maximization is assumed.
 *
 * The software in this file is the result of (ongoing) scientific research.
 * The software has been constructed based on
 * Linkage Tree Genetic Algorithm (LTGA) and
 * Multi-objective Adapted Maximum-Likelihood Gaussian Model Iterated Density 
 * Estimation Evolutionary Algorithm (MAMaLGaM)
 *
 * Interested readers can refer to the following publications for more details:
 *
 * 1. N.H. Luong, H. La Poutré, and P.A.N. Bosman: Multi-objective Gene-pool 
 * Optimal Mixing Evolutionary Algorithms with the Interleaved Multi-start Scheme. 
 * In Swarm and Evolutionary Computation, vol. 40, June 2018, pages 238-254, 
 * Elsevier, 2018.
 * 
 * 2. N.H. Luong, H. La Poutré, and P.A.N. Bosman: Multi-objective Gene-pool 
 * Optimal Mixing Evolutionary Algorithms. In Dirk V. Arnold, editor,
 * Proceedings of the Genetic and Evolutionary Computation Conference GECCO 2014: 
 * pages 357-364, ACM Press New York, New York, 2014.
 *
 * 3. P.A.N. Bosman and D. Thierens. More Concise and Robust Linkage Learning by
 * Filtering and Combining Linkage Hierarchies. In C. Blum and E. Alba, editors,
 * Proceedings of the Genetic and Evolutionary Computation Conference -
 * GECCO-2013, pages 359-366, ACM Press, New York, New York, 2013. 
 *
 * 4. P.A.N. Bosman. The anticipated mean shift and cluster registration 
 * in mixture-based EDAs for multi-objective optimization. In M. Pelikan and
 * J. Branke, editors, Proceedings of the Genetic and Evolutionary Computation 
 * GECCO 2010, pages 351-358, ACM Press, New York, New York, 2010.
 * 
 * 5. J.C. Pereira, F.G. Lobo: A Java Implementation of Parameter-less 
 * Evolutionary Algorithms. CoRR abs/1506.08694 (2015)
 *
 * In addition, the code, as provided here, was modified for the following project:
 * DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
 *
 * This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
 *
 * Project leaders: Peter A.N. Bosman, Tanja Alderliesten
 * Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
 * Main code developer: Arthur Guijt 
 */

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Includes -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include <sys/time.h>
//
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <cmath>
#include <random>
#include <string>
#include <sstream>
#include <tuple>
#include <vector>
#include <map>
#include <variant>
#include <algorithm>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cassert>
#include <numeric>
#include <chrono>
//
#include <rectangular_lsap.h>
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-= Section Header Functions -=-=-=-=-=-=-=-=-=-=-=-=*/
/*---------------------------- Utility Functions ---------------------------*/
void *Malloc( long size );
void initializeRandomNumberGenerator();
double randomRealUniform01( void );
int randomInt( int maximum );
double log2( double x ) noexcept;
int* createRandomOrdering(int size_of_the_set);
double distanceEuclidean( double *x, double *y, int number_of_dimensions, double* ranges );

int *mergeSort( double *array, int array_size );
void mergeSortWithinBounds( double *array, int *sorted, int *tosort, int p, int q );
void mergeSortMerge( double *array, int *sorted, int *tosort, int p, int r, int q );
/*-------------------------Interpret Command Line --------------------------*/
void interpretCommandLine( int argc, char **argv );
void parseCommandLine( int argc, char **argv );
void parseOptions( int argc, char **argv, int *index );
void printAllInstalledProblems( void );
void optionError( char **argv, int index );
void parseParameters( int argc, char **argv, int *index );
void printUsage( void );
void checkOptions( void );
void printVerboseOverview( void );
/*--------------- Load Problem Data and Solution Evaluations ---------------*/
void evaluateIndividual(char *solution, double *obj, double *con, void* metadata, int objective_index_of_extreme_cluster);
char *installedProblemName( int index );
int numberOfInstalledProblems( void );

void onemaxLoadProblemData();
void trap5LoadProblemData();
void lotzLoadProblemData();
void onemaxProblemEvaluation(char *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster);
double deceptiveTrapKTightEncodingFunctionProblemEvaluation( char *parameters, int k, char is_one );
void trap5ProblemEvaluation(char *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster);
void lotzProblemEvaluation(char *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster);

void knapsackLoadProblemData();
void ezilaitiniKnapsackProblemData();
void knapsackSolutionRepair(char *solution, double *solution_profits, double *solution_weights, double *solution_contraint, int objective_index_of_extreme_cluster);
void knapsackSolutionSingleObjectiveRepair(char *solution, double *solution_profits, double *solution_weights, double *solution_constraint, int objective_index);
void knapsackSolutionMultiObjectiveRepair(char *solution, double *solution_profits, double *solution_weights, double *solution_constraint);
void knapsackProblemEvaluation(char *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster);

void maxcutLoadProblemData();
void ezilaitiniMaxcutProblemData();
void maxcutReadInstanceFromFile(char *filename, int objective_index);
void maxcutProblemEvaluation( char *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster );

double **getDefaultFrontOnemaxZeromax( int *default_front_size );
double **getDefaultFrontTrap5InverseTrap5( int *default_front_size );
double **getDefaultFrontLeadingOneTrailingZero( int *default_front_size );
short haveDPFSMetric( void );
double **getDefaultFront( int *default_front_size );
double computeDPFSMetric( double **default_front, int default_front_size, double **approximation_front, int approximation_front_size );
/*---------------------------- Tracking Progress ---------------------------*/
void writeGenerationalStatistics();
void writeCurrentElitistArchive( char final );
void logElitistArchiveAtSpecificPoints();
char checkTerminationCondition();
char checkPopulationTerminationCriterion();
char checkNumberOfEvaluationsTerminationCondition();
char checkVTRTerminationCondition();
void logNumberOfEvaluationsAtVTR();
/*---------------------------- Elitist Archive -----------------------------*/
char isDominatedByElitistArchive( double *obj, double con, char *is_new_nondominated_point, int *position_of_existed_member );
short sameObjectiveBox( double *objective_values_a, double *objective_values_b );
int hammingDistanceInParameterSpace(char *solution_1, char *solution_2);
int hammingDistanceToNearestNeighborInParameterSpace(char *solution, int replacement_position);
void updateElitistArchive( char *solution, double *solution_objective_values, double solution_constraint_value, void* metadata );
void updateElitistArchiveWithReplacementOfExistedMember( char *solution, double *solution_objective_values, double solution_constraint_value, void* solution_metadata, char *is_new_nondominated_point, char *is_dominated_by_archive);
void removeFromElitistArchive( int *indices, int number_of_indices );
short isInListOfIndices( int index, int *indices, int number_of_indices );
void addToElitistArchive( char *solution, double *solution_objective_values, double solution_constraint_value, void* metadata);
void adaptObjectiveDiscretization( void );
/*-------------------------- Solution Comparision --------------------------*/
char betterFitness( double *objective_value_x, double constraint_value_x, double *objective_value_y, double constraint_value_y, int objective_index );
char equalFitness( double *objective_value_x, double constraint_value_x, double *objective_value_y, double constraint_value_y, int objective_index );
short constraintParetoDominates( double *objective_values_x, double constraint_value_x, double *objective_values_y, double constraint_value_y );
short constraintWeaklyParetoDominates( double *objective_values_x, double constraint_value_x, double *objective_values_y, double constraint_value_y );
short paretoDominates( double *objective_values_x, double *objective_values_y );
short weaklyParetoDominates( double *objective_values_x, double *objective_values_y );
/*-------------------------- Linkage Tree Learning --------------------------*/
void learnLinkageTree( int cluster_index );
double *estimateParametersForSingleBinaryMarginal(  int cluster_index, int *indices, int number_of_indices, int *factor_size );
int determineNearestNeighbour( int index, double **S_matrix, int mpm_length );
void printLTStructure( int cluster_index );
/*-------------------------------- MO-GOMEA --------------------------------*/
void initialize();
void initializeMemory();
void initializePopulationAndFitnessValues();
void computeObjectiveRanges( void );

void learnLinkageOnCurrentPopulation();
int** clustering(double **objective_values_pool, int pool_size, int number_of_dimensions, 
                    int number_of_clusters, int *pool_cluster_size, double** objective_means_scaled );
int *greedyScatteredSubsetSelection( double **points, int number_of_points, int number_of_dimensions, int number_to_select );
int *greedyScatteredSubSubsetSelection( std::vector<int> &indices, double **points, int number_of_dimensions, int number_to_select, double* ranges );
void determineExtremeClusters(int number_of_mixing_components, double** objective_means_scaled, int* which_extreme);
void initializeClusters();
void ezilaitiniClusters();

void improveCurrentPopulation( void );
void copyValuesFromDonorToOffspring(char *solution, char *donor, int cluster_index, int linkage_group_index);
void copyFromAToB(char *solution_a, double *obj_a, double con_a, void* &metadata_a, char *solution_b, double *obj_b, double *con_b, void* &metadata_b);
void copyFromAToBSubset(char *solution_a, double *obj_a, double con_a, void* &metadata_a, char *solution_b, double *obj_b, double *con_b, void* &metadata_b, int *subset, int subset_size);
void mutateSolution(char *solution, int lt_factor_index, int cluster_index);
void performMultiObjectiveGenepoolOptimalMixing( int cluster_index, int solution_index, char *parent, double *parent_obj, double parent_con, void *parent_metadata,
                                            char *result, double *obj, double *con, void *metadata );
void performSingleObjectiveGenepoolOptimalMixing( int cluster_index, int objective_index, int solution_idx,
                                char *parent, double *parent_obj, double parent_con, void *parent_metadata,
                                char *result, double *obj, double *con, void *metadata);

void selectFinalSurvivors();
void freeAuxiliaryPopulations();
void performUHVIObjectiveGenepoolOptimalMixing(
    int cluster_index, int solution_index, double **pseudo_objective_values, char *parent, double *parent_obj,
    double parent_con, void *parent_metadata, char *result, double *obj, double *con,
    void *metadata);
void performRankHVObjectiveGenepoolOptimalMixing(
    int cluster_index, int solution_index, double **pseudo_objective_values, char *parent, double *parent_obj,
    double parent_con, void *parent_metadata, char *result, double *obj, double *con,
    void *metadata);
void performLineMultiObjectiveGenepoolOptimalMixing(
    int cluster_index, int line_index, int solution_idx, char *parent,
    double *parent_obj, double parent_con, void *parent_metadata,
    char *result, double *obj, double *con, void *metadata);

/*-------------------------- Parameter-less Scheme -------------------------*/
void initializeMemoryForArrayOfPopulations();
void putInitializedPopulationIntoArray();
void assignPointersToCorrespondingPopulation();
void ezilaitiniArrayOfPopulation();
void ezilaitiniMemoryOfCorrespondingPopulation();
void schedule_runMultiplePop_clusterPop_learnPop_improvePop();
void schedule();

void initializeCommonVariables();
void ezilaitiniCommonVariables();
void loadProblemData();
void ezilaitiniProblemData();
void run();
int main( int argc, char **argv );
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=- Section Global Variables -=-=-=-=-=-=-=-=-=-=-=-=-*/
char    **population,               /* The population containing the solutions. */
        ***array_of_populations,    /* The array containing all populations in the parameter-less scheme. */
        **offspring,                /* Offspring solutions. */
        **elitist_archive,          /* Archive of elitist solutions. */
        **elitist_archive_copy;     /* Copy of the elitist archive. */

int     problem_index,                          /* The index of the optimization problem. */
        number_of_parameters,                   /* The number of parameters to be optimized. */
        number_of_generations,                  /* The current generation count. */
        *array_of_number_of_generations,        /* The array containing generation counts of all populations in the parameter-less scheme.*/
        generation_base,                        /* The number of iterations that population of size N_i is run before running 1 iteration of population of size N_(i+1). */
        population_size,                        /* The size of each population. */
        *array_of_population_sizes,             /* The array containing population sizes of all populations in the parameter-less scheme. */
        smallest_population_size,               /* The size of the first population. */
        population_id,                          /* The index of the population that is currently operating. */
        offspring_size,                         /* The size of the offspring population. */

        number_of_objectives,                   /* The number of objective functions. */
        elitist_archive_size,                   /* Number of solutions in the elitist archive. */
        elitist_archive_size_target,            /* The lower bound of the targeted size of the elitist archive. */
        elitist_archive_copy_size,              /* Number of solutions in the elitist archive copy. */
        elitist_archive_capacity,               /* Current memory allocation to elitist archive. */
        number_of_mixing_components,            /* The number of components in the mixture distribution. */
        orig_number_of_mixing_components,       /* When overriden, the original number of mixture components. -1 if not overridden. */
        *array_of_number_of_clusters,           /* The array containing the number-of-cluster of each population in the parameter-less scheme. */
        *population_cluster_sizes,              /* The size of each cluster. */
        **population_indices_of_cluster_members,/* The corresponding index in the population of each solution in a cluster. */
        *which_extreme,                         /* The corresponding objective of an extreme-region cluster. */
        
        t_NIS,                          /* The number of subsequent generations without an improvement (no-improvement-stretch). */
        *array_of_t_NIS,                /* The array containing the no-improvement-stretch of each population in the parameter-less scheme. */
        **array_of_s_NIS,               /* The array containing the no-improvement-stretch for each solution in each population. */
        *s_NIS,                         /* The array containing the no-improvement-stretch for each solution in the current generation. */
        maximum_number_of_populations,  /* The maximum number of populations that can be run (depending on memory budget). */
        number_of_populations,          /* The number of populations that have been initialized. */

        **mpm,                          /* The marginal product model. */
        *mpm_number_of_indices,         /* The number of variables in each factor in the mpm. */
        mpm_length,                     /* The number of factors in the mpm. */

        ***lt,                          /* The linkage tree, one for each cluster. */
        **lt_number_of_indices,         /* The number of variables in each factor in the linkage tree of each cluster. */
        *lt_length;                     /* The number of factors in the linkage tree of each cluster. */
	      
long    number_of_evaluations,            /* The current number of times a function evaluation was performed. */
        log_progress_interval,            /* The interval (in terms of number of evaluations) at which the elitist archive is logged. */
		maximum_number_of_evaluations,    /* The maximum number of evaluations. */
        *array_of_number_of_evaluations_per_population; /* The array containing the number of evaluations used by each population in the parameter-less scheme. */
		
double  **objective_values,                 /* Objective values for population members. */
        ***array_of_objective_values,       /* The array containing objective values of all populations in the parameter-less scheme. */
        *constraint_values,                 /* Constraint values of population members. */
        **array_of_constraint_values,       /* The array containing constraint values of all populations in the parameter-less scheme. */
        
        **objective_values_offspring,       /* Objective values of offspring solutions. */
        *constraint_values_offspring,       /* Constraint values of offspring solutions. */
                 
        **elitist_archive_objective_values,         /* Objective values of solutions stored in elitist archive. */
        **elitist_archive_copy_objective_values,    /* Copy of objective values of solutions stored in elitist archive. */
        *elitist_archive_constraint_values,         /* Constraint values of solutions stored in elitist archive. */
        *elitist_archive_copy_constraint_values,    /* Copy of constraint values of solutions stored in elitist archive. */

        *objective_ranges,                          /* Ranges of objectives observed in the current population. */
        **array_of_objective_ranges,                /* The array containing ranges of objectives observed in each population in the parameter-less scheme. */
        **objective_means_scaled,                   /* The means of the clusters in the objective space, linearly scaled according to the observed ranges. */
        *objective_discretization,                  /* The length of the objective discretization in each dimension (for the elitist archive). */
        vtr,                              /* The value-to-reach (in terms of Inverse Generational Distance). */
        **MI_matrix;                      /* Mutual information between any two variables */

int64_t random_seed = 0,                      /* The seed used for the random-number generator. */
        random_seed_changing = 0;             /* Internally used variable for randomly setting a random seed. */

char    use_pre_mutation,                   /* Whether to use weak mutation. */
        use_pre_adaptive_mutation,          /* Whether to use strong mutation. */
        use_print_progress_to_screen,       /* Whether to print the progress of the optimization to screen. */
        use_repair_mechanism,               /* Whether to use a repair mechanism (provided by users) if the problem is constrained. */
        *optimization,                      /* Maximization or Minimization for each objective. */
        print_verbose_overview,             /* Whether to print a overview of settings (0 = no). */
        use_vtr,                            /* Whether to terminate at the value-to-reach (VTR) (0 = no). */
        objective_discretization_in_effect, /* Whether the objective space is currently being discretized for the elitist archive. */
        elitist_archive_front_changed;      /* Whether the Pareto front formed by the elitist archive is changed in this generation. */


std::chrono::time_point<std::chrono::system_clock> time_at_last_archive_improvement;
long number_of_evaluations_at_last_archive_improvement = 0;

// k stored across generations
int **array_of_solution_k,
     *solution_k;

// Metadata: extra associated data for each solution.

void cleanupSolutionMetadata(void* &metadata);
void initSolutionMetadata(void* &metadata);
void copySolutionMetadata(void* &metadata_from, void* &metadata_to);

// returns true if any headers were written (i.e. do we need to add a comma afterwards?).
bool writeCSVHeaderMetadata(std::ostream &outstream, bool has_preceding_field);
// returns true if any fields were written (i.e. do we need to add a comma afterwards?).
bool writeCSVMetadata(std::ostream &outstream, void* metadata, bool has_preceding_field);

// For each population, an array of void pointers
void***  array_of_solution_metadatas;
// Array of void pointers (one for each population member).
void**   solution_metadata;
void**   solution_metadata_offspring;

void**   elitist_archive_solution_metadata;
void**   elitist_archive_copy_solution_metadata;

bool use_metadata = true;

bool write_offspring_halfway_generation = true;
bool keep_only_latest_elitist_archive = true;
bool keep_only_latest_population = true;
bool keep_only_latest_lines = true;

// 1 - Change
// 2 - Improvement
// 4 - Replacement
// Or any combination using |
char reset_nis_on = 2 | 4;

int initialized_population_size;
int initialized_offspring_size;

std::chrono::time_point<std::chrono::system_clock> time_at_start;

// Date
bool allow_starting_of_more_than_one_population;

double*** array_of_tsch_weight_vectors; /* Storage for the direction vectors of each population */
double** tsch_weight_vectors; /* Direction vectors of current population */
void performTschebysheffObjectiveGenepoolOptimalMixing(
    int cluster_index, int solution_index, 
    char *parent, double *parent_obj, double parent_con, void *parent_metadata,
    char *result, double *obj, double *con, void *metadata,
    double *tsch_weight_vector, double* reference, double* ranges);

int smallest_number_of_clusters;
bool fixed_number_of_clusters = false;

std::vector<double> objective_range;
std::vector<double> objective_min;
std::vector<double> objective_max;

std::vector<double> archive_objective_min;
std::vector<double> archive_objective_max;

std::vector<double> utopian_point;
std::vector<double> nadir_point;

double adjusted_nadir_adjustment = 0.05;
std::vector<double> adjusted_nadir;

std::vector<std::vector<double>> objective_tcheb_weights;

void computeObjectiveMinMaxRangesNadirAndUtopianPoint();

bool write_hypervolume = true;
bool write_weights_to_file = false;
bool write_elitist_archive = true;
bool write_population = false;
bool write_statistics_dat_file = true;

bool write_clusters_with_statistics = false;
bool write_initial_population = false;
bool write_last_archive_improvement_always = true;

bool use_solution_NIS_instead_of_population_NIS = true;

// 0 - MI
// 1 - NMI
// 2 - Filtered MI
// 3 - Filtered NMI
// -1 - Random
int linkage_mode = 3;

// New problems
std::optional<std::string> instance;

#define MAXSAT 5
std::vector<std::vector<int>> maxsat_instance;
std::optional<int> maxsat_vtr;
std::vector<std::vector<char>> maxsat_attractor;
void maxsatProblemEvaluation( char *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster );

#define MAXCUT_VS_ONEMAX 6
void maxcutVsOnemaxProblemEvaluation( char *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster );

#define BESTOFTRAPS 7
void bestOfTrapsProblemEvaluation( char *solution, double *obj_values, double *con_value, void* metadata, int /* objective_index_of_extreme_cluster */ );

#define BESTOFTRAPS_VS_ONEMAX 8
void bestOfTrapsVsOneMaxProblemEvaluation( char *solution, double *obj_values, double *con_value, void* metadata, int /* objective_index_of_extreme_cluster */ );

#define BESTOFTRAPS_VS_MAXCUT 9
void bestOfTrapsVsMaxCutProblemEvaluation( char *solution, double *obj_values, double *con_value, void* metadata, int /* objective_index_of_extreme_cluster */ );

#define DISCRETIZED_CONTINOUS_PROBLEM 10
void DCProblemEvaluation( char *solution, double *obj_values, double *con_value, int /* objective_index_of_extreme_cluster */ );

// Front override
std::optional<std::filesystem::path> overrideFrontPath;
void loadOverrideFront();

// Vector containing the lines, after they have been determined.
// First part of the pair is the back (worst), while the latter is the front (best).
std::vector<std::pair<std::vector<double>, std::vector<double>>> lines;
std::vector<std::vector<double>> lines_direction_normalized;
std::vector<int> nearest_line;

std::vector<int> rank_order;
std::vector<int> rank;
std::vector<std::pair<int, int>> rank_start_end;
int max_rank;

std::vector<double> hv_per_rank;
std::vector<double> hv_contributions_per_population_member;

//  0 - None
//  1 - Has extreme objective (i.e. is most extreme)
//  2 - When ordering by an objective, the solution has an objective value
//      better than (1 - extreme_radius) * |P| solutions.
//  3 - Same as 2, but sqrt(|P|) solutions.
//  4 - Same as 2, but 2 * sqrt(|P|) solutions.
//  5 - Per rank - has extreme objective.
//  6 - Per rank - is within ceil(sqrt(|R|)) of best solutions for an objective within a rank.
//  7 - The solution has an objective value better than (1 - extreme_radius) * |P| solutions.
//      AND is one of ceil(sqrt(|P|)) solutions selected by GSS based on hamming distance.
//  8 - Per rank - The solution has an objective value better than (1 - extreme_radius) * |R| solutions.
//      AND is one of ceil(sqrt(|R|)) solutions selected by GSS based on hamming distance.
//  9 - Is KNN of solution with extreme objective.
// 10 - Is locally extreme within the KNN graph.
//      Where local is identified by having no overlap in the KNN of previously
//      visited nodes.
int extreme_kernel_mode = 0;
double extreme_radius = 0.05;
std::vector<int> is_extreme_kernel;
void determineExtremeKernels();

// Whether to operate the hypervolume computation in a steady state style.
bool hv_steady_state = false;
// Whether to use a neighborhood to compute improvement HV over.
bool hv_improvement_subset = false;
// Whether to enable subsets for use in forced improvement.
bool hv_improvement_subset_fi = true;
// Whether to correct for the density around a particular frontal solution.
bool hv_density_correction = false;
std::vector<double> kernel_density_estimate;
void updateCurrentKernelDensityEstimate();

// Whether to use donor search.
bool use_donor_search = true;

// -1 = Off
//  0 = Default (Select randomly from archive)
// 1 = Nearest Line Neighbor in archive.
int fi_mode = 0;
bool fi_do_replacement = true;
// 0 - no
// 1 - selector only
// 2 - replacement only
// 3 - yes
char fi_use_singleobjective = 3;

std::vector<std::vector<int>> objective_ordered_reference_points;
void determineORPs();

// -1: Current Objective provides direction
// -2: New Objective provides direction
int kernel_improvement_mode = -1;


#define CLUSTER_MODE_DEFAULT 0
#define CLUSTER_MODE_LINE 1
#define CLUSTER_MODE_LINE_KNN 2
#define CLUSTER_MODE_COMPUTE_LINES_ONLY 3
#define CLUSTER_MODE_HAMMING_GSS 4
#define CLUSTER_MODE_HAMMING_INV_HAMMING_GSS 5
#define CLUSTER_MODE_HAMMING_KNN_GSS 6
#define CLUSTER_MODE_HAMMING_HIERARCHICAL_KNN_GSS 7
#define CLUSTER_MODE_HAMMING_KNN_ALL 8
#define CLUSTER_MODE_HAMMING_KNN_SYM_ALL 9
#define CLUSTER_MODE_HAMMING_FILTERKNN_ALL 10
#define CLUSTER_MODE_HAMMING_FILTERKNN_SYM_ALL 11

// For each member of the population, this array can be used
// to pre-assign mixtures to solutions.
// (for example in a cluster-for-each-solution setting.)
std::vector<int> mixture_assignment;

// Clustering mode, only impacts original GOMEA.
//
// 0. default - standard clustering approach (KMeans)
// 1. lines - cluster using lines.
int clustering_mode = CLUSTER_MODE_DEFAULT;
int clustering_mode_modifier = 0;
bool domination_mode_uses_clustering_mode = true;
int domination_based_clustering_mode = CLUSTER_MODE_DEFAULT;

// 0 - No local searcher
// 1 - Random Direction Hillclimber
// 2 - Starting Direction Hillclimber
// 3 - Nadir -> Current Solution Direction Linear Hillclimber
// 4 - Nadir -> New Solution Direction Linear Hillclimber
// 5 - Preassigned Weights
// 6 - Preassigned Weights with Exchange
int initialization_ls_mode = 0;
bool multipass = true;

int nondominated_sort(int n_points, double** points, int* rank, 
    std::optional<std::reference_wrapper<std::vector<int>>> order,
    std::optional<std::reference_wrapper<std::vector<std::pair<int, int>>>> order_start_stop);

// 0 - Euclidean Clustering
// 1 - Line Clustering
// 2 - Line Clustering & Direction Kernel
// 3 - Line Clustering & Direction Kernel, Disabled Classical Domination step.
// 4 - Like 1, but use_single_objective_directions = false
// 5 - Like 2, but use_single_objective_directions = false
// 6 - Line Clustering & Kernel: Direction & Neighborhood (KNN: Euclidean)
// 7 - Line Clustering & Kernel: Direction & Neighborhood (KNN: Line)
// 8 - Like 6, but use_single_objective_directions = false
// 9 - Like 7, but use_single_objective_directions = false
//
//  -1 - Hoang's Scalarization MO-GOMEA
//  -2 - Line Clustering (Duplicate)
//  -3 - Hoang's Scalarization MO-GOMEA with Line Clustering
//  -4 - Rank HV MO-GOMEA - Euclidean Clustering
//  -5 - Rank HV MO-GOMEA - Line Clustering
//  -6 - Steady State Rank HV MO-GOMEA - Euclidean Clustering
//  -7 - Steady State Rank HV MO-GOMEA - Line Clustering
//  -8 - UHVI MO-GOMEA - Euclidean Clustering
//  -9 - UHVI MO-GOMEA - Line Clustering
// -10 - Steady State UHVI MO-GOMEA - Euclidean Clustering
// -11 - Steady State UHVI MO-GOMEA - Line Clustering
int approach_kind = 0;

// 0 - Original MO-GOMEA
// 1 - Domination/Assigned-Line MO-GOMEA
// 2 - Domination/Kernel-Line MO-GOMEA
// 3 - Line Only MO-GOMEA
// 4 - Rank HV MO-GOMEA, using an acceptation criterion similar to:
//      - Wang, Hao, André Deutz, Thomas Bäck, and Michael Emmerich. 2017. 
//         ‘Hypervolume Indicator Gradient Ascent Multi-Objective Optimization’.
//        In Evolutionary Multi-Criterion Optimization, 654–69. 
//        Lecture Notes in Computer Science. Cham: Springer International Publishing.
//        https://doi.org/10.1007/978-3-319-54157-0_44.
// 5 - UHVI MO-GOMEA, using an acceptation criterion based on the UHVI indicator described in:
//      - Touré, Cheikh, Nikolaus Hansen, Anne Auger, and Dimo Brockhoff. 2019. 
//         ‘Uncrowded Hypervolume Improvement: COMO-CMA-ES and the Sofomore Framework’.
//        In Proceedings of the Genetic and Evolutionary Computation Conference, 638–46. GECCO ’19. 
//        New York, NY, USA: Association for Computing Machinery.
//        https://doi.org/10.1145/3321707.3321852.
int approach_mode = 0;

// 0 - Nearest-Line
// 1 - Assign-Equally-Minimizing-Distance
int line_assignment_mode = 0;

int arg_population_size = -1;
int arg_num_clusters = 0;
// 0 - default, positive is fixed size, negative is starting size
// 1 - population size based kind. 
char arg_num_cluster_mode = 0;

bool log_mixing_invocations = false;
bool log_agreements_with_nearest_when_assigning_lines = false;

// TODO: Make these flags of interest
bool normalize_objectives_for_lines = true;
bool preserve_extreme_frontal_points = false;
bool use_knn_per_lineinstead_of_assigned_line = true;
bool use_cluster_line_instead_of_assigned_line_for_line_mixing = false;
bool use_single_objective_directions = true;
bool use_nadir_of_population = true;
bool use_original_clustering_with_domination_iteration = true;

bool do_normalize_hypervolume = true;

bool ignore_identical_solutions_for_knn = true;

bool use_scalarization = false;

bool terminate_population_upon_convergence = true;

// Neighborhood mode - How to select the solution
// 0 - Default: Cluster Index
// 1 - KNN (Euclidean)
// 2 - KNN (Line)
// The following approaches are based on a change on perspective, rather than using the KNN
// of yourself, we use of whom you are a KNN.
// Directly this would yield an approach that has some issues,
// most notably you may not neccesarily be a nearest neighbor to anyone.
// Hence order solutions by our solutions rank in their order.
// Note that for speed these are computed at the start of a generation.
// Can be kernelized by sending out 'rank updates'.
// 3 - Other KNN (Euclidean)
// 4 - Other KNN (Line)
// 5 - KNN (Euclidean) with 2k + Reverse Hamming Distance (k)
// 6 - KNN (Line) with 2k + Reverse Hamming Distance (k)
// 7 - KNN (Euclidean) with 2k + Reverse FOS Distance (k)
// 8 - KNN (Line) with 2k + Reverse FOS Distance (k)
// 9 - KNN Hamming Distance
// 10 - KNN (Euclidean) with 2k + Hamming Distance (k)
// 11 - KNN (Line) with 2k + Hamming Distance (k)
// 12 - KNN (Hamming Distance) with 2k + Euclidean (k)
// 13 - KNN (Hamming Distance) with 2k + Line (k)
int mixing_pool_mode = 0;

// The number of nearest neighbors to be used for KNN.
int mixing_pool_knn_k_mode = -1;

std::vector<int> assigned_line;

// 0 - Best & Worst
// 1 - Best & Worst, Fallback to nadir for worst.
// 2 - Best & Always use nadir for worst point.
// 3 - Normalized Any, Always use nadir for worst point.
int line_mode = 3;
// Whether to align the nearest line to axis.
bool align_nearest_lines_to_axis = true;

// Line damping factor, 0.0 is use current, 1.0 is use assigned previous point. In between is averaged.
double damping_factor = 0.0;

std::filesystem::path outpath = ".";

// MAXCUT Problem Variables
int     ***maxcut_edges, 
        *number_of_maxcut_edges;
double  **maxcut_edges_weights;
// Knapsack Problem Variables
double  **profits,
        **weights,
        *capacities,
        *ratio_profit_weight;
int     *item_indices_least_profit_order;
int     **item_indices_least_profit_order_according_to_objective;
/*------------------- Termination of Smaller Populations -------------------*/
char    *array_of_population_statuses;
double  ***array_of_Pareto_front_of_each_population;
int     *array_of_Pareto_front_size_of_each_population;
char    stop_population_when_front_is_covered;
void updateParetoFrontForCurrentPopulation(double **objective_values_pop, double *constraint_values_pop, int pop_size);
void checkWhichSmallerPopulationsNeedToStop();
char checkParetoFrontCover(int pop_index_1, int pop_index_2);
void ezilaitiniArrayOfParetoFronts();
void initializeArrayOfParetoFronts();
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/* Variables and types related to override the reference front. */
struct OverrideFront
{
    int number_of_objectives;
    int number_of_points;
    std::optional<std::string> instance; // For checking.
    bool has_reference_point;

    std::vector<double> objectives;
};

// std::optional<std::filesystem::path> overrideFrontPath;
std::optional<OverrideFront> overrideFront;

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Constants -=-=-=-=-=-=-=-=-=-=-=-=-=-*/
#define FALSE 0
#define TRUE 1

#define NOT_EXTREME_CLUSTER -1

#define MINIMIZATION 1
#define MAXIMIZATION 2

#define ZEROMAX_ONEMAX 0
#define TRAP5 1
#define KNAPSACK 2
#define LOTZ 3
#define MAXCUT 4
/*-=-=-=-=-=-=-=-=-=-=-=-= Section Utility Function -=-=-=-=-=-=-=-=-=-=-=-*/
/**
 * Allocates memory and exits the program in case of a memory allocation failure.
 */
void *Malloc( long size )
{
    void *result;

    result = (void *) malloc( size );
    if( !result )
    {
        printf("\n");
        printf("Error while allocating memory in Malloc( %ld ), aborting program.", size);
        printf("\n");

        throw std::runtime_error("Failure to allocate");
        exit( 1 );
    }

    return( result );
}
/**
 * Initializes the pseudo-random number generator.
 */
void initializeRandomNumberGenerator()
{
    struct timeval tv;
    struct tm *timep;

    if (random_seed == 0)
    {

        while( random_seed_changing == 0 )
        {
            gettimeofday( &tv, NULL );
            timep = localtime (&tv.tv_sec);
            random_seed_changing = timep->tm_hour * 3600 * 1000 + timep->tm_min * 60 * 1000 + timep->tm_sec * 1000 + tv.tv_usec / 1000;
        }

        random_seed = random_seed_changing;
    }
    else
    {
        random_seed_changing = random_seed;
    }
}
/**
 * Returns a random double, distributed uniformly between 0 and 1.
 */
double randomRealUniform01( void )
{
    int64_t n26, n27;
    double  result;

    random_seed_changing = (random_seed_changing * 0x5DEECE66DLLU + 0xBLLU) & ((1LLU << 48) - 1);
    n26                  = (int64_t)(random_seed_changing >> (48 - 26));
    random_seed_changing = (random_seed_changing * 0x5DEECE66DLLU + 0xBLLU) & ((1LLU << 48) - 1);
    n27                  = (int64_t)(random_seed_changing >> (48 - 27));
    result               = (((int64_t)n26 << 27) + n27) / ((double) (1LLU << 53));

    return( result );
}
        
/**
 * Returns a random integer, distributed uniformly between 0 and maximum.
 */
int randomInt( int maximum )
{
    int result;
    result = (int) (((double) maximum)*randomRealUniform01());
    return( result );
}
/**
 * Computes the two-log of x.
 */
double math_log_two = log(2.0);
double log2( double x ) noexcept
{
  return( log(x) / math_log_two );
}
int* createRandomOrdering(int size_of_the_set)
{
    int *order, a, b, c, i;

    order = (int *) Malloc( size_of_the_set*sizeof( int ) );
    for( i = 0; i < size_of_the_set; i++ )
        order[i] = i;
    for( i = 0; i < size_of_the_set; i++ )
    {
        a        = randomInt( size_of_the_set );
        b        = randomInt( size_of_the_set );
        c        = order[a];
        order[a] = order[b];
        order[b] = c;
    }

    return order;
}
void setVectorToRandomOrdering(std::vector<int> &v)
{
    int *random_order = createRandomOrdering(v.size());
    std::copy(random_order, random_order + v.size(), v.begin());
    free(random_order);
}
/**
 * Computes the Euclidean distance between two points.
 */
double distanceEuclidean( double *x, double *y, int number_of_dimensions, double* ranges = NULL )
{
    int    i;
    double value, result;

    result = 0.0;
    for( i = 0; i < number_of_dimensions; i++ )
    {
        value   = y[i] - x[i];
        if (ranges != NULL)
            value /= ranges[i];
        result += value*value;
    }
    result = sqrt( result );

    return( result );
}
/**
 * Sorts an array of doubles and returns the sort-order (small to large).
 */
int *mergeSort( double *array, int array_size )
{
    int i, *sorted, *tosort;

    sorted = (int *) Malloc( array_size * sizeof( int ) );
    tosort = (int *) Malloc( array_size * sizeof( int ) );
    for( i = 0; i < array_size; i++ )
        tosort[i] = i;

    if( array_size == 1 )
        sorted[0] = 0;
    else
        mergeSortWithinBounds( array, sorted, tosort, 0, array_size-1 );

    free( tosort );

    return( sorted );
}
/**
 * Subroutine of merge sort, sorts the part of the array between p and q.
 */
void mergeSortWithinBounds( double *array, int *sorted, int *tosort, int p, int q )
{
    int r;

    if( p < q )
    {
        r = (p + q) / 2;
        mergeSortWithinBounds( array, sorted, tosort, p, r );
        mergeSortWithinBounds( array, sorted, tosort, r+1, q );
        mergeSortMerge( array, sorted, tosort, p, r+1, q );
    }
}
/**
 * Subroutine of merge sort, merges the results of two sorted parts.
 */
void mergeSortMerge( double *array, int *sorted, int *tosort, int p, int r, int q )
{
    int i, j, k, first;

    i = p;
    j = r;
    for( k = p; k <= q; k++ )
    {
        first = 0;
        if( j <= q )
        {
            if( i < r )
            {
                if( array[tosort[i]] < array[tosort[j]] )
                first = 1;
            }
        }
        else
            first = 1;

        if( first )
        {
            sorted[k] = tosort[i];
            i++;
        }
        else
        {
            sorted[k] = tosort[j];
            j++;
        }
    }

    for( k = p; k <= q; k++ )
        tosort[k] = sorted[k];
}

/**
 * Find a reference point for computing the hypervolume from.
 * 
 * Uses the approach published in [1].
 *
 * [1] Nowak, Krzysztof, Marcus Märtens, and Dario Izzo. 2014. ‘Empirical Performance of the Approximation of the Least Hypervolume Contributor’.
 *     In Parallel Problem Solving from Nature – PPSN XIII, edited by Thomas Bartz-Beielstein, Jürgen Branke, Bogdan Filipič, and Jim Smith, 662–71.
 *     Lecture Notes in Computer Science. Cham: Springer International Publishing.
 *     https://doi.org/10.1007/978-3-319-10762-2_65.
 */
std::vector<double> getReferencePoint(int number_of_objectives, double* nadir, double* ideal, double alpha=0.01)
{
    std::vector<double> hv_reference_point(number_of_objectives);

    for (int d = 0; d < number_of_objectives; ++d)
    {
        double diff_nadir_ideal = (nadir[d] - ideal[d]);
        // Note: nadir & ideal are assumed to have taken the optimization direction into account already.
        hv_reference_point[d] = nadir[d] + alpha * diff_nadir_ideal;
    }

    return hv_reference_point;
}

bool updateAdjustedNadir(char* direction, double* ideal, double* adjusted_nadir, double* new_point, int dimensionality)
{
    bool changed = false;
    for (int d = 0; d < dimensionality; ++d)
    {
        int dir = (direction[d] == MAXIMIZATION) ? 1 : -1;
        double range_if_nadir = std::abs(ideal[d] - new_point[d]);
        double new_adjusted_point_d = new_point[d] - range_if_nadir * dir * adjusted_nadir_adjustment;
            
        if (dir * adjusted_nadir[d] > dir * new_adjusted_point_d)
        {
            // If current point contributed the nadir, the nadir worsens.
            // update adjusted nadir.
            adjusted_nadir[d] = new_adjusted_point_d;
            changed = true;
        }
    }
    return changed;
}

/**
 * Computes the hypervolume of a set of 2D points.
 *
 * Using a simple algorithm for computing 2D hypervolumes by sorting and performing a single sweep.
 *
 * @param direction A 2 element-long array indentifying whether the respective dimension is supposed to be maximized (2) or minimized (1).
 */
double compute2DHypervolume(int number_of_points, char* direction, double* reference_point, double** points)
{
    // Shortcut:
    // Hypervolume for an empty set of points is defined to be zero.
    if (number_of_points == 0) return 0.0;
    
    std::vector<int> indices(number_of_points);
    std::iota(indices.begin(), indices.end(), 0);

    // Sort the points such that the first objective is 'worsening'.
    // In case of equality: do the same for the second objective.
    int m0 = (direction[0] == MAXIMIZATION) ? 1 : -1;
    int m1 = (direction[1] == MAXIMIZATION) ? 1 : -1;
    std::sort(indices.begin(), indices.end(), 
        [&points, m0, m1](int a, int b)
        {
            auto point_a = std::make_pair(m0 * points[a][0], m1 * points[a][1]);
            auto point_b = std::make_pair(m0 * points[b][0], m1 * points[b][1]);
            return point_a > point_b;  
        }
    );

    // Filter out any points with values worse than the reference point: they do not contribute anything.
    if (number_of_points > 0)
    {
        auto new_end = std::remove_if(indices.begin(), indices.end(), [points, reference_point, m0, m1](int i) {
            bool is_worse_than_reference_in_obj0 = m0 * points[i][0] < m0 * reference_point[0];
            bool is_worse_than_reference_in_obj1 = m1 * points[i][1] < m1 * reference_point[1];
            return is_worse_than_reference_in_obj0 || is_worse_than_reference_in_obj1;
        });
        int new_length = std::distance(indices.begin(), new_end);
        indices.resize(new_length);
        number_of_points = indices.size();
    }

    // Filter out dominated points. If list is of size one, the point is undominated.
    // Furthermore: exclude points that are worse than the reference point in an objective.
    // None of these points should contribute to the hypervolume.
    if (number_of_points > 0)
    {
        // As we have the first objective sorted such that it is worsening.
        // And the second objective sorted such that it is worsening if the first objective is equal.
        // We can prune dominated points easily: any undominated point should have an ascending second objective.
        double o2 = points[indices[0]][1];
        auto new_end = std::remove_if(indices.begin() + 1, indices.end(), 
            [&o2, points, m1](int i)
            {
                if (m1 * o2 < m1 * points[i][1])
                {
                    // Improved on second objective and the first encountered.
                    o2 = points[i][1];

                    // Keep!
                    return false;
                }
                else
                {
                    // Worse in both objectives: is dominated.
                    return true;
                }
            }
        );
        int new_length = std::distance(indices.begin(), new_end);
        indices.resize(new_length);
    }
    
    // Update the number of points left.
    number_of_points = indices.size();
    // If after pruning there are no points left HV = 0.0.
    if (number_of_points == 0) return 0.0;
    
    // Initial hypervolume is the square for the first point.
    double hv = std::abs(points[indices[0]][0] - reference_point[0]) * std::abs(points[indices[0]][1] - reference_point[1]);


    // Other contributions.
    for (int indices_idx = 1; indices_idx < number_of_points; ++indices_idx)
    {
        int idx = indices[indices_idx];
        int prev_idx = indices[indices_idx - 1];

        // Much like the first point, each point has their own volume with the reference point.
        double idx_volume = std::abs(points[idx][0] - reference_point[0]) * std::abs(points[idx][1] - reference_point[1]);
        
        // We do however have to avoid counting overlap twice (or more!)
        // By having sorted the indices & knowing they are pareto optimal
        // we know the following which we can make use of:
        // - points[idx - 1][0] > points[idx][0]
        // - points[idx - 1][1] < points[idx][1]
        // Furthermore: these facts hold across the entire line.
        // meaning that the first coordinate will be monotonically increasing,
        // whereas the second coordinate will be monotonically decreasing.
        //
        // This provides the insight that allows for easy avoidance of double counting.
        //
        // See the following diagram, where X & Y represent the point with index `prev_idx` and `idx`
        // respectively, and where R is the reference point. Assuming maximization.
        //
        //       .-----X
        //       |  A  |
        // OBJ 0 |-----------Y
        //       |  C  |  B  |
        //       R-----------.
        //           OBJ 1
        //
        // The area marked with A will not be intersected by future indices,
        // and is the contribution of point X (barring the previous subtraction).
        // Area C is mutual between the both of them, and needs to be subtracted,
        // as otherwise it is counted twice.
        // B is the potential contribution of Y - excluding any subtractions happening.
        // As we are not interested in the contributions themselves (right now)
        // We simply compute the area of C, which is the difference between R, and the point
        // formed by taking the X coordinate of X, and the Y coordinate of Y.
        double overlap = std::abs(points[idx][0] - reference_point[0]) * std::abs(points[prev_idx][1] - reference_point[1]);

        // Update the running total of hypervolume.
        hv += idx_volume - overlap;
    }

    return hv;
}

/**
 * Computes the hypervolume of a set of 2D points.
 *
 * Using a simple algorithm for computing 2D hypervolumes by sorting and performing a single sweep.
 *
 * @param direction A 2 element-long array indentifying whether the respective dimension is supposed to be maximized (2) or minimized (1).
 */
double compute2DHypervolumeSubsetAndContributions(
    int* subset, int number_of_points_in_subset, char* direction, double* reference_point, double** points, double* contributions)
{
    // Shortcut:
    // Hypervolume for an empty set of points is defined to be zero.
    if (number_of_points_in_subset == 0) return 0.0;
    
    std::vector<int> indices(number_of_points_in_subset);
    std::copy(subset, subset + number_of_points_in_subset, indices.begin());
    // Initialize contributions for subset to zero.
    if (contributions != NULL)
        for (int i : indices)
            contributions[i] = 0.0;

    int number_of_points = number_of_points_in_subset;

    // Sort the points such that the first objective is 'worsening'.
    // In case of equality: do the same for the second objective.
    int m0 = (direction[0] == MAXIMIZATION) ? 1 : -1;
    int m1 = (direction[1] == MAXIMIZATION) ? 1 : -1;
    std::sort(indices.begin(), indices.end(), 
        [&points, m0, m1](int a, int b)
        {
            auto point_a = std::make_pair(m0 * points[a][0], m1 * points[a][1]);
            auto point_b = std::make_pair(m0 * points[b][0], m1 * points[b][1]);
            return point_a > point_b;  
        }
    );

    // Filter out any points with values worse than the reference point: they do not contribute anything.
    // As contributions has been initialized to zero. We don't need to set the contributions here.
    if (number_of_points > 0)
    {
        auto new_end = std::remove_if(indices.begin(), indices.end(), [points, reference_point, m0, m1](int i) {
            bool is_worse_than_reference_in_obj0 = m0 * points[i][0] < m0 * reference_point[0];
            bool is_worse_than_reference_in_obj1 = m1 * points[i][1] < m1 * reference_point[1];
            return is_worse_than_reference_in_obj0 || is_worse_than_reference_in_obj1;
        });
        int new_length = std::distance(indices.begin(), new_end);
        indices.resize(new_length);
        number_of_points = indices.size();
    }

    // Filter out dominated points. If list is of size one, the point is undominated.
    // Furthermore: exclude points that are worse than the reference point in an objective.
    // None of these points should contribute to the hypervolume.
    // As contributions has been initialized to zero. We don't need to set the contributions here either.
    if (number_of_points > 0)
    {
        // As we have the first objective sorted such that it is worsening.
        // And the second objective sorted such that it is worsening if the first objective is equal.
        // We can prune dominated points easily: any undominated point should have an ascending second objective.
        double o2 = points[indices[0]][1];
        auto new_end = std::remove_if(indices.begin() + 1, indices.end(), 
            [&o2, points, m1](int i)
            {
                if (m1 * o2 < m1 * points[i][1])
                {
                    // Improved on second objective and the first encountered.
                    o2 = points[i][1];

                    // Keep!
                    return false;
                }
                else
                {
                    // Worse in both objectives: is dominated.
                    return true;
                }
            }
        );
        int new_length = std::distance(indices.begin(), new_end);
        indices.resize(new_length);
    }
    
    // Update the number of points left.
    number_of_points = indices.size();
    // If after pruning there are no points left HV = 0.0.
    if (number_of_points == 0) return 0.0;
    
    // Initial hypervolume is the square for the first point.
    double hv = std::abs(points[indices[0]][0] - reference_point[0]) * std::abs(points[indices[0]][1] - reference_point[1]);
    contributions[indices[0]] = hv;

    // Other contributions.
    for (int indices_idx = 1; indices_idx < number_of_points; ++indices_idx)
    {
        int idx = indices[indices_idx];
        int prev_idx = indices[indices_idx - 1];

        // Much like the first point, each point has their own volume with the reference point.
        double idx_volume = std::abs(points[idx][0] - reference_point[0]) * std::abs(points[idx][1] - reference_point[1]);
        contributions[idx] = idx_volume;
        
        // We do however have to avoid counting overlap twice (or more!)
        // By having sorted the indices & knowing they are pareto optimal
        // we know the following which we can make use of:
        // - points[idx - 1][0] > points[idx][0]
        // - points[idx - 1][1] < points[idx][1]
        // Furthermore: these facts hold across the entire line.
        // meaning that the first coordinate will be monotonically increasing,
        // whereas the second coordinate will be monotonically decreasing.
        //
        // This provides the insight that allows for easy avoidance of double counting.
        //
        // See the following diagram, where X & Y represent the point with index `prev_idx` and `idx`
        // respectively, and where R is the reference point. Assuming maximization.
        //
        //       .-----X
        //       |  A  |
        // OBJ 0 |-----------Y
        //       |  C  |  B  |
        //       R-----------.
        //           OBJ 1
        //
        // The area marked with A will not be intersected by future indices,
        // and is the contribution of point X (barring the previous subtraction).
        // Area C is mutual between the both of them, and needs to be subtracted,
        // as otherwise it is counted twice.
        // B is the potential contribution of Y - excluding any subtractions happening.
        // As we are not interested in the contributions themselves (right now)
        // We simply compute the area of C, which is the difference between R, and the point
        // formed by taking the X coordinate of X, and the Y coordinate of Y.
        double overlap = std::abs(points[idx][0] - reference_point[0]) * std::abs(points[prev_idx][1] - reference_point[1]);

        // Update the running total of hypervolume.
        hv += idx_volume - overlap;

        // We do not assign the overlap as a contribution to any of the points.
        contributions[idx] -= overlap;
        contributions[prev_idx] -= overlap;
        // But there may have been overlap between the two subtractions that has happened twice.
        // Take the following figure.
        //
        //       .-----X
        //       |  A  |
        // OBJ 0 |-----------Y
        //       |  D  |  B  |
        //       |-----------------Z
        //       |  E  |  F  |  C  |
        //       R-----------------.
        //           OBJ 1
        //
        // The contribution for Y subtracts region E twice:
        // Once due to the overlap with X. Once due to the overlap with Z.
        if (indices_idx > 1)
        {
            int prev_prev_idx = indices[indices_idx - 2];
            double overlap_overlap = std::abs(points[idx][0] - reference_point[0]) * std::abs(points[prev_prev_idx][1] - reference_point[1]);
            contributions[prev_idx] += overlap_overlap;
        }
    }

    return hv;
}

/**
 * Computes the hypervolume of a set of 2D points.
 *
 * Using a simple algorithm for computing 2D hypervolumes by sorting and performing a single sweep.
 * Computes the UHVI for each solution in subset. (For all undominated solutions, their HV contribution is stored anyways.)
 *
 * @param direction A 2 element-long array indentifying whether the respective dimension is supposed to be maximized (2) or minimized (1).
 */
double compute2DUncrowdedHypervolumeSubsetContributions(
    int* subset_for_contributions, int number_of_points_in_subset,
    int* indices_to_use, int number_of_points, char* direction, double* reference_point, double** points, double* contributions,
    bool do_density_correction, double* ranges = NULL)
{
    // Shortcut:
    // Hypervolume for an empty set of points is defined to be zero.
    if (number_of_points == 0) return 0.0;

    // int number_of_initial_points = number_of_points;

    std::vector<int> indices(indices_to_use, &indices_to_use[number_of_points - 1]);
    // std::iota(indices.begin(), indices.end(), 0);
    // Initialize contributions for subset to zero.
    if (contributions != NULL)
        for (int i : indices)
            contributions[i] = 0.0;

    int highest_idx = 0;
    for (int i : indices)
        highest_idx = std::max(i, highest_idx);

    // Sort the points such that the first objective is 'worsening'.
    // In case of equality: do the same for the second objective.
    int m0 = (direction[0] == MAXIMIZATION) ? 1 : -1;
    int m1 = (direction[1] == MAXIMIZATION) ? 1 : -1;
    std::sort(indices.begin(), indices.end(), 
        [&points, m0, m1](int a, int b)
        {
            auto point_a = std::make_pair(m0 * points[a][0], m1 * points[a][1]);
            auto point_b = std::make_pair(m0 * points[b][0], m1 * points[b][1]);
            return point_a > point_b;  
        }
    );

    // Filter out any points with values worse than the reference point: they do not contribute anything.
    // As contributions has been initialized to zero. We don't need to set the contributions here.
    if (number_of_points > 0)
    {
        auto new_end = std::remove_if(indices.begin(), indices.end(), [points, reference_point, m0, m1](int i) {
            bool is_worse_than_reference_in_obj0 = m0 * points[i][0] < m0 * reference_point[0];
            bool is_worse_than_reference_in_obj1 = m1 * points[i][1] < m1 * reference_point[1];
            return is_worse_than_reference_in_obj0 || is_worse_than_reference_in_obj1;
        });
        int new_length = std::distance(indices.begin(), new_end);
        indices.resize(new_length);
        number_of_points = indices.size();
    }

    // Filter out dominated points. If list is of size one, the point is undominated.
    // Furthermore: exclude points that are worse than the reference point in an objective.
    // None of these points should contribute to the hypervolume.
    // As contributions has been initialized to zero. We don't need to set the contributions here either.
    if (number_of_points > 0)
    {
        // As we have the first objective sorted such that it is worsening.
        // And the second objective sorted such that it is worsening if the first objective is equal.
        // We can prune dominated points easily: any undominated point should have an ascending second objective.
        int ox = indices[0];
        double o2 = points[ox][1];
        auto new_end = std::remove_if(indices.begin() + 1, indices.end(), 
            [&ox, &o2, points, m1](int i)
            {
                if (m1 * o2 < m1 * points[i][1])
                {
                    // Improved on second objective and the first encountered.
                    ox = i;
                    o2 = points[i][1];

                    // Keep!
                    return false;
                }
                else
                {
                    // Worse in both objectives: is dominated.
                    return true;
                }
            }
        );
        int new_length = std::distance(indices.begin(), new_end);
        indices.resize(new_length);
    }
    
    // Update the number of points left.
    number_of_points = indices.size();
    // If after pruning there are no points left HV = 0.0.
    if (number_of_points == 0) return 0.0;
    
    // Initial hypervolume is the square for the first point.
    double hv = std::abs(points[indices[0]][0] - reference_point[0]) * std::abs(points[indices[0]][1] - reference_point[1]);
    contributions[indices[0]] = hv;

    // Other contributions.
    for (int indices_idx = 1; indices_idx < number_of_points; ++indices_idx)
    {
        int idx = indices[indices_idx];
        int prev_idx = indices[indices_idx - 1];

        // Much like the first point, each point has their own volume with the reference point.
        double idx_volume = std::abs(points[idx][0] - reference_point[0]) * std::abs(points[idx][1] - reference_point[1]);
        contributions[idx] = idx_volume;
        
        // We do however have to avoid counting overlap twice (or more!)
        // By having sorted the indices & knowing they are pareto optimal
        // we know the following which we can make use of:
        // - points[idx - 1][0] > points[idx][0]
        // - points[idx - 1][1] < points[idx][1]
        // Furthermore: these facts hold across the entire line.
        // meaning that the first coordinate will be monotonically increasing,
        // whereas the second coordinate will be monotonically decreasing.
        //
        // This provides the insight that allows for easy avoidance of double counting.
        //
        // See the following diagram, where X & Y represent the point with index `prev_idx` and `idx`
        // respectively, and where R is the reference point. Assuming maximization.
        //
        //       .-----X
        //       |  A  |
        // OBJ 0 |-----------Y
        //       |  C  |  B  |
        //       R-----------.
        //           OBJ 1
        //
        // The area marked with A will not be intersected by future indices,
        // and is the contribution of point X (barring the previous subtraction).
        // Area C is mutual between the both of them, and needs to be subtracted,
        // as otherwise it is counted twice.
        // B is the potential contribution of Y - excluding any subtractions happening.
        // As we are not interested in the contributions themselves (right now)
        // We simply compute the area of C, which is the difference between R, and the point
        // formed by taking the X coordinate of X, and the Y coordinate of Y.
        double overlap = std::abs(points[idx][0] - reference_point[0]) * std::abs(points[prev_idx][1] - reference_point[1]);

        // Update the running total of hypervolume.
        hv += idx_volume - overlap;

        // We do not assign the overlap as a contribution to any of the points.
        contributions[idx] -= overlap;
        contributions[prev_idx] -= overlap;
        // But there may have been overlap between the two subtractions that has happened twice.
        // Take the following figure.
        //
        //       .-----X
        //       |  A  |
        // OBJ 0 |-----------Y
        //       |  D  |  B  |
        //       |-----------------Z
        //       |  E  |  F  |  C  |
        //       R-----------------.
        //           OBJ 1
        //
        // The contribution for Y subtracts region E twice:
        // Once due to the overlap with X. Once due to the overlap with Z.
        if (indices_idx > 1)
        {
            int prev_prev_idx = indices[indices_idx - 2];
            double overlap_overlap = std::abs(points[idx][0] - reference_point[0]) * std::abs(points[prev_prev_idx][1] - reference_point[1]);
            contributions[prev_idx] += overlap_overlap;
        }
        // If the contribution is not greater than or equal to zero, the solution should have been removed during the domination step.
        // (equality happens if the point is exactly on the border)
        assert(contributions[prev_idx] >= 0);
    }
    // int number_of_points_for_each_frontal_point = (number_of_initial_points + number_of_points - 1) / number_of_points;
    int i_last = indices.size() - 1;
    for (int ii = 0; ii < number_of_points_in_subset; ++ii)
    {
        int i = subset_for_contributions[ii];
        if (contributions[i] > 0.0) continue;
        
        // i is dominated and does not contribute anything. UHVI should be computed!
        // There are a few settings, but they are roughly laid out like this:
        //
        //  [ x - undominated point / part of front] [ i - current point ]
        //         v(a)
        //       |----x (1)
        //       |    |<(b) v(c)
        //       |    L------x (2)
        // OBJ 0 |           | 
        //       |           |<(d) v(e)
        //       |  i        L------x (3)
        //       L__________________| < (f)
        //               OBJ 1
        //
        // UHVI for i is equal to the closest point on the front, which is the euclidean distance from
        // i to any point on the line segments. These line segments are bounded by the points x, but also by
        // the in-between points L, for the first L_1 = (x(x_1), y(x_2)), L_2 = (x(x_2), y(x_3))
        //
        // This routine solves this problem by determining the distances to the outer edges,
        // and then the nearest point on each pair of line fragments.
        // The latter we can do because the point i is dominated (otherwise we wouldn't be here!):
        // this means one of the nearest coordinates must correspond to L, which is also a point on the other line.
        
        // So first: Outer edges.
        double da = (points[indices[0]][0]      - points[i][0]) / (ranges == NULL ? 1 : ranges[0]);
        double db = (points[indices[i_last]][1] - points[i][1]) / (ranges == NULL ? 1 : ranges[1]);
        double distance = std::min(
            da,
            db
        );
        int x_i_best = da < db ? indices[0] : indices[i_last];
        int x_j_best = x_i_best;
        // Sequential Pairs
        // PERFORMANCE: Rather than looping over all points (as done here), the start and end points can be found.
        for (size_t idx = 0; idx < indices.size() - 1; ++idx)
        {
            int x_i = indices[idx];
            int x_j = indices[idx + 1];

            double nearest_point_pair[2] = {
                (optimization[0] == MAXIMIZATION) ? std::max(points[x_i][0], points[i][0]) : std::min(points[x_i][0], points[i][0]),
                (optimization[1] == MAXIMIZATION) ? std::max(points[x_j][1], points[i][1]) : std::min(points[x_j][1], points[i][1])
            };

            double new_euclidean_distance = distanceEuclidean(points[i], nearest_point_pair, 2, ranges);

            if (new_euclidean_distance < distance)
            {
                distance = new_euclidean_distance;
                x_i_best = x_i;
                x_j_best = x_j;
            }
        }
        if (do_density_correction)
        {
            double adj_distance = distance / (kernel_density_estimate[x_i_best] + kernel_density_estimate[x_j_best] + 2.0);
            // The resulting contribution is then -distance.
            contributions[i] = -adj_distance;
        }
        else
        {
            contributions[i] = -distance;
        }
    }

    return hv;
}

void updateCurrentKernelDensityEstimate()
{
    std::vector<int> indices(population_size);
    std::vector<int> hamming(population_size);
    int k = std::ceil(std::sqrt(population_size));
    kernel_density_estimate.resize(population_size);
    for (int i = 0; i < population_size; ++i)
    {
        for (int j = 0; j < population_size; ++j)
        {
            hamming[j] = hammingDistanceInParameterSpace(population[i], population[j]);
        }
        auto hamming_from_i = [&hamming](int a, int b)
        {
            int dia = hamming[a];
            int dib = hamming[b];
            return dia < dib;
        };
        std::nth_element(indices.begin(), indices.begin() + k, indices.end(), hamming_from_i);
        kernel_density_estimate[i] = static_cast<double>(hamming[indices[k]]) / static_cast<double>(number_of_parameters);
    }
}

std::vector<double> hv_reference_point;
double hv_default_front = 0.0;

void determineReferencePointForCurrentProblemHV()
{
    
    int default_front_size = 0;

    // Note: all default fronts are stored in a static variable.
    // This means they don't need to get free'd.

    double** default_front = getDefaultFront( &default_front_size );

    if (default_front_size <= 0)
        return;

    if (overrideFront.has_value() && overrideFront->has_reference_point)
    {
        // Copy over the reference point instead, if provided.
        hv_reference_point.resize(overrideFront->number_of_objectives);
        std::copy(
            overrideFront->objectives.end() - overrideFront->number_of_objectives,
            overrideFront->objectives.end(),
            hv_reference_point.begin());
    }
    else
    {
        std::vector<double> default_front_nadir(number_of_objectives);
        std::vector<double> default_front_ideal(number_of_objectives);

        // Initialize.
        for (int d = 0; d < number_of_objectives; ++d)
        {
            default_front_nadir[d] = default_front[0][d];
            default_front_ideal[d] = default_front[0][d];
        }

        // Determine nadir & ideal of front.
        for (int f_idx = 1; f_idx < default_front_size; ++f_idx)
        {
            for (int d = 0; d < number_of_objectives; ++d)
            {
                default_front_nadir[d] = (optimization[d] == MAXIMIZATION) ?
                    std::min(default_front_nadir[d], default_front[f_idx][d]) :
                    std::max(default_front_nadir[d], default_front[f_idx][d]);
                default_front_ideal[d] = (optimization[d] == MAXIMIZATION) ?
                    std::max(default_front_ideal[d], default_front[f_idx][d]) :
                    std::min(default_front_ideal[d], default_front[f_idx][d]);
            }
        }

        // Compute reference point
        hv_reference_point = getReferencePoint(number_of_objectives, default_front_nadir.data(), default_front_ideal.data());
    }
    // Compute default front HV
    hv_default_front = compute2DHypervolume(default_front_size, optimization, hv_reference_point.data(), default_front);
}

bool canComputeHypervolume()
{
    return static_cast<int>(hv_reference_point.size()) == number_of_objectives;
}
double computeCurrentHypervolume()
{
    // No valid reference point. As such hypervolume can not be computed.
    assert(canComputeHypervolume());

    double hv = compute2DHypervolume(elitist_archive_size, optimization, hv_reference_point.data(), elitist_archive_objective_values);

    if (do_normalize_hypervolume)
        return hv / hv_default_front;
    else
        return hv;
}

/**
 * For Tchebysheff scalarized GOMEA similar to the approach described in
 *   "Luong, Ngoc Hoang, Tanja Alderliesten, and Peter A. N. Bosman. 2018.
 *   ‘Improving the Performance of MO-RV-GOMEA on Problems with Many Objectives Using Tchebycheff Scalarizations’.
 *   In Proceedings of the Genetic and Evolutionary Computation Conference, 705–12. GECCO ’18. 
 *   New York, NY, USA: Association for Computing Machinery. https://doi.org/10.1145/3205455.3205498."
 *
 * Note that unlike the paper above, this implementation is targeted towards binary (or discrete valued) problems
 * rather than real-valued problems.
 **/

/**
 * Returns a matrix of `number_of_directions` `number_of_dimensions`-dimensional vectors, where for each vector holds that their values sum to 1.
 **/
double** computeRandomPositiveDirections(int number_of_directions, int number_of_dimensions)
{
    double** result = (double**) Malloc(number_of_directions * sizeof(double*));

    for (int i = 0; i < number_of_directions; ++i)
    {
        result[i] = (double*) Malloc(number_of_dimensions * sizeof(double));
        double length = 0.0;
        for (int d = 0; d < number_of_dimensions; ++d)
        {
            double v = randomRealUniform01();
            result[i][d] = v;
            length += v; // * v;
        }
        // length = std::sqrt(length);
        for (int d = 0; d < number_of_dimensions; ++d)
        {
            result[i][d] = result[i][d] / length;
        }

    }
    return result;
}

void normalizeDirectionsOrder2(double** directions, int number_of_directions, int number_of_dimensions)
{
    for (int i = 0; i < number_of_directions; ++i)
    {
        double l = 0.0;
        for (int d = 0; d < number_of_dimensions; ++d)
        {
            l += directions[i][d] * directions[i][d];
        }
        l = std::sqrt(l);
        for (int d = 0; d < number_of_dimensions; ++d)
        {
            directions[i][d] = directions[i][d] / l;
        }

    }
}

/**
 * Returns a matrix of `number_of_directions` `number_of_dimensions`-dimensional vectors,
 * where for each vector holds that their values sum to 1, and such that the vectors are approximately
 * nicely spread out.
 **/
double** computeApproximatelyEquidistantPositiveDirections(int number_of_directions, int number_of_dimensions)
{
    // Constants / Parameters
    int number_of_points_for_greedy_subset_scattering = std::max(100000, 2 * number_of_directions);
    bool include_axis_aligned_vectors = true;

    double** result = (double**) Malloc(number_of_directions * sizeof(double*));
    int starting_index = 0;
    if (include_axis_aligned_vectors)
    {
        // If this statement is false, axis alignment will be an issue as the number of directions is insufficient
        // to contain all axis-aligned directions
        assert(number_of_directions >= number_of_dimensions);

        for (int d_i = 0; d_i < number_of_dimensions; ++d_i)
        {
            result[d_i] = (double*) Malloc(number_of_dimensions * sizeof(double));
            std::fill(result[d_i], result[d_i] + number_of_dimensions, 0.0);
            result[d_i][d_i] = 1.0;
        }

        // Ensure that we do not override the axis aligned vectors.
        starting_index = number_of_dimensions;
    }

    // Generate random directions to sample equidistant points from
    double **random_directions = computeRandomPositiveDirections(number_of_points_for_greedy_subset_scattering, number_of_dimensions);
    // 
    int number_of_points_to_select = number_of_directions - starting_index;
    int* indices = greedyScatteredSubsetSelection(random_directions, number_of_points_for_greedy_subset_scattering, number_of_dimensions, number_of_points_to_select);
    
    // Copy over selected points to result.
    for (int i_result = 0; i_result < number_of_points_to_select; ++i_result)
    {
        double* current = (double*) Malloc(number_of_dimensions * sizeof(double));
        int i_random = indices[i_result];
        std::copy(random_directions[i_random], random_directions[i_random] + number_of_dimensions, current);
        result[starting_index + i_result] = current;
    }

    // Free the memory used.
    free(indices);
    for (int i = 0; i < number_of_points_for_greedy_subset_scattering; ++i)
        free(random_directions[i]);
    free(random_directions);

    if (write_weights_to_file)
    {
        std::ofstream file;
        file.open(outpath / ("weights_" + std::to_string(population_size) + ".dat"), std::ios::out);
        
        for (int i = 0; i < number_of_directions; ++i)
        {
            for (int d = 0; d < number_of_dimensions; ++d)
            {
                file << result[i][d];
                if (d == number_of_dimensions - 1)
                    file << "\n";
                else
                    file << ",";
            }
        }
        file.close();
    }
    return result;
}

void computeRangesAndIdeal(int number_of_dimensions, char* direction, int population_size, double** objective_values,
    double* &ranges, double* &ideal)
{
    // Compute ranges and ideal point.
    ranges = (double*) Malloc(number_of_dimensions * sizeof(double));
    ideal = (double*) Malloc(number_of_dimensions * sizeof(double));
    for (int d = 0; d < number_of_dimensions; ++d)
    {
        double min = INFINITY;
        double max = -INFINITY;
        for (int i = 0; i < population_size; ++i)
        {
            min = std::min(min, objective_values[i][d]);
            max = std::max(max, objective_values[i][d]);
        }
        ranges[d] = max - min;
        ideal[d] = (direction[d] == MAXIMIZATION) ? max : min;
    }
}

double computeTchebysheffDistance(int number_of_dimensions, double* point, double* reference, double* weights, double* ranges = NULL)
{
    double distance = -INFINITY;
    for (int d = 0; d < number_of_dimensions; ++d)
    {
        double difference_d = std::abs(point[d] - reference[d]);
        double norm_difference_d = difference_d / ((ranges == NULL) ? 1 : ranges[d]);
        double weighted = weights[d] * norm_difference_d;
        distance = std::max(distance, weighted);
    }
    return distance;
}

int findWeightWithSmallestTchebysheffDistance(int number_of_dimensions, double *point, double *reference,
                                              int number_of_weight_vectors, double **weight_vectors,
                                              int *indices = NULL, double *ranges = NULL)
{
    // Initialize.
    int result = 0;
    if (indices != NULL) result = indices[0];
    double result_distance = INFINITY;
    // Find weight vector with the smallest Tschebysheff distance for this point.
    for (int idx = 0; idx < number_of_weight_vectors; ++idx)
    {
        int w_i = (indices == NULL) ? idx : indices[idx];
        double distance = computeTchebysheffDistance(number_of_dimensions, point, reference, weight_vectors[w_i], ranges);

        if (distance < result_distance)
        {
            result = idx;
            result_distance = distance;
        }
    }
    return result;
}

/**
 * Create an assignment of solutions to weight vectors by assigning each solution to the weight vector
 * with the smallest index.
 * If indices is not NULL it will return the index in the indices array instead.
 **/
int* scalarizedAssignWeightVectors(int number_of_dimensions, char* direction, int population_size, double** objective_values, int number_of_directions, double** directions)
{
    // Compute ranges and ideal point.
    double* ranges; double* ideal;
    computeRangesAndIdeal(number_of_dimensions, direction, population_size, objective_values, ranges, ideal);

    // Assign each population member to the weight vector with the smallest weight.
    int* result = (int*) Malloc( sizeof(int) * number_of_dimensions );
    for (int i = 0; i < population_size; ++i)
    {
        result[i] = findWeightWithSmallestTchebysheffDistance(number_of_dimensions, objective_values[i], ideal, number_of_directions, directions, NULL, ranges);
    }
    return result;
}


int findPointWithSmallestTchebysheffDistance(int number_of_dimensions, double *weight_vector, double *reference,
                                              int number_of_points, double **points,
                                              int *indices = NULL, double *ranges = NULL)
{
    // Initialize.
    int result = 0;
    double result_distance = INFINITY;
    // Find population item with the smallest Tschebysheff distance for this point.
    for (int idx = 0; idx < number_of_points; ++idx)
    {
        int p_i = (indices == NULL) ? idx : indices[idx];
        double distance = computeTchebysheffDistance(number_of_dimensions, points[p_i], reference, weight_vector, ranges);

        if (distance < result_distance)
        {
            result = idx;
            result_distance = distance;
        }
    }
    return result;
}

int* scalarizedAssignWeightVectorsEvenly(int number_of_dimensions, char* optimization_direction, int population_size, double** objective_values, int number_of_directions, double** directions, double* &ranges, double* &ideal)
{
    // Implementation currently requires the number of directions to be equal to the population size.
    // As otherwise we would have to assign a vector more than once.
    assert(population_size == number_of_directions);
    
    // Compute ranges and ideal point.
    computeRangesAndIdeal(number_of_dimensions, optimization_direction, population_size, objective_values, ranges, ideal);

    //
    int* result = (int*) Malloc( sizeof(int) * population_size );

    // Keep track of which points have already been assigned.
    std::vector<int> indices(number_of_directions);
    setVectorToRandomOrdering(indices);
    int number_of_points_left = population_size;

    // Visit weights in a random order.
    int* weight_order = createRandomOrdering(population_size);
    for (int i_o = 0; i_o < number_of_directions; ++i_o)
    {
        int i = weight_order[i_o];
        // Find index in indices of weight with smallest distance.
        int indices_idx = findPointWithSmallestTchebysheffDistance(number_of_dimensions, directions[i], ideal, number_of_points_left, objective_values, indices.data(), ranges);
        int point_idx = indices[indices_idx];
        result[point_idx] = i;
        // Shrink part that is scanned by `findWeightWithSmallestTchebysheffDistance`
        number_of_points_left -= 1;
        // Remove used direction by replacing with the index that should not be removed (but is no longer part of the list).
        indices[indices_idx] = indices[number_of_points_left];
    }
    free(weight_order);

    return result;
}

/*-=-=-=-=-=-=-=-=-=-=- Section Interpret Command Line -=-=-=-=-=-=-=-=-=-=-*/
/**
 * Parses and checks the command line.
 */
void interpretCommandLine( int argc, char **argv )
{
    parseCommandLine( argc, argv );
  
    checkOptions();
}
/**
 * Parses the command line.
 * For options, see printUsage.
 */
void parseCommandLine( int argc, char **argv )
{
    int index;

    index = 1;

    parseOptions( argc, argv, &index );
  
    parseParameters( argc, argv, &index );
}

int approach_arg1 = 0;

/**
 * Processes the approach_kind, and alters flags & its own value where neccesary
 * to describe various behaviors.
 */
void processApproachMode()
{
    switch (approach_kind)
    {
    case -22:
        // KNN Steady-state DUHV
        // hv_improvement_subset = true;
        hv_density_correction = true;
        hv_steady_state = true;
        approach_mode = 5;
        break;
    case -21:
        // KNN Steady-state SDUHV
        hv_improvement_subset = true;
        hv_density_correction = true;
        hv_steady_state = true;
        approach_mode = 5;
        break;
    case -20:
        // KNN Steady-state SDUHV
        hv_improvement_subset = true;
        hv_density_correction = true;
        // hv_steady_state = true;
        approach_mode = 5;
        break;
    case -19:
        // KNN Steady-state UHV
        hv_improvement_subset = true;
        hv_steady_state = true;
        approach_mode = 5;
        break;
    case -18:
        // KNN UHV
        hv_improvement_subset = true;
        // hv_steady_state = true;
        approach_mode = 5;
        break;
    case -17:
        // KNN Steady-state RankHV
        hv_improvement_subset = true;
        hv_steady_state = true;
        approach_mode = 4;
        break;
    case -16:
        // KNN RankHV
        hv_improvement_subset = true;
        hv_steady_state = true;
        approach_mode = 4;
        break;
    case -15:
        // Domination Based & Steady-state UHV
        approach_mode = 7;
        hv_steady_state = true;
        domination_mode_uses_clustering_mode = false;
        break;
    case -14:
        // Domination Based & UHV
        approach_mode = 7;
        domination_mode_uses_clustering_mode = false;
        break;
    case -13:
        // Domination Based & Steady-state RankHV
        hv_steady_state = true;
        approach_mode = 6;
        domination_mode_uses_clustering_mode = false;
        break;
    case -12:
        // Domination Based & RankHV
        approach_mode = 6;
        domination_mode_uses_clustering_mode = false;
        break;
    case -11:
        // Steady-state UHV / Line clustering
        hv_steady_state = true;
        approach_mode = 5;
        clustering_mode = CLUSTER_MODE_LINE;
        break;
    case -10:
        // Steady-state UHV
        hv_steady_state = true;
        approach_mode = 5;
        break;
    case -9:
        // UHV / Line clustering
        approach_mode = 5;
        clustering_mode = CLUSTER_MODE_LINE;
        break;
    case -8:
        // UHV
        approach_mode = 5;
        break;
    case -7:
        // Steady-state RankHV / Line clustering
        hv_steady_state = true;
        approach_mode = 4;
        clustering_mode = CLUSTER_MODE_LINE;
        break;
    case -6:
        // Steady-state RankHV
        hv_steady_state = true;
        approach_mode = 4;
        break;
    case -5:
        // RankHV / Line clustering
        approach_mode = 4;
        clustering_mode = CLUSTER_MODE_LINE;
        break;
    case -4:
        // RankHV
        approach_mode = 4;
        break;
    case -3:
        // Scalarized MO-GOMEA but we use line clustering.
        approach_mode = 0;
        clustering_mode = CLUSTER_MODE_LINE;
        use_scalarization = true;
        break;
    case -2:
        // MO-GOMEA but we use line clustering.
        approach_mode = 0;
        clustering_mode = CLUSTER_MODE_LINE;
        break;
    case -1:
        // Scalarized MO-GOMEA
        approach_mode = 0;
        use_scalarization = true;
        break;
    case 4: case 5:
        // Remapping for approach 1 & 2 where use_single_objective_directions = false.
        approach_mode = approach_kind - 3;
        use_single_objective_directions = false;
        break;

    case 6:
        approach_mode = 2;
        mixing_pool_mode = 1;
        mixing_pool_knn_k_mode = approach_arg1;
        break;
    case 7:
        approach_mode = 2;
        mixing_pool_mode = 2;
        mixing_pool_knn_k_mode = approach_arg1;
        break;
    case 8:
        approach_mode = 2;
        mixing_pool_mode = 3;
        mixing_pool_knn_k_mode = approach_arg1;
        break;
    case 9:
        approach_mode = 2;
        mixing_pool_mode = 4;
        mixing_pool_knn_k_mode = approach_arg1;
        break;
    case 10:
        approach_mode = 2;
        mixing_pool_mode = 5;
        mixing_pool_knn_k_mode = approach_arg1;
        break;
    case 11:
        approach_mode = 2;
        mixing_pool_mode = 6;
        mixing_pool_knn_k_mode = approach_arg1;
        break;
    case 12:
        approach_mode = 2;
        mixing_pool_mode = 7;
        mixing_pool_knn_k_mode = approach_arg1;
        break;   
    case 13:
        approach_mode = 2;
        mixing_pool_mode = 8;
        mixing_pool_knn_k_mode = approach_arg1;
        break;
    case 14:
        approach_mode = 1;
        line_assignment_mode = 1;
        break;
    case 15:
        approach_mode = 1;
        line_assignment_mode = 1;
        mixing_pool_mode = 7;
        mixing_pool_knn_k_mode = approach_arg1;
        break;
    case 42:
        // Approach that uses kernels only for directional accept.
        approach_mode = 2;
        mixing_pool_mode = 0;
        break;

    // case 10:
    //     approach_mode = 2;
    //     mixing_pool_mode = 1;
    //     mixing_pool_knn_k_mode = approach_arg1;
    //     use_single_objective_directions = false;
    //     break;
    // case 10:
    //     approach_mode = 2;
    //     mixing_pool_mode = 2;
    //     mixing_pool_knn_k_mode = approach_arg1;
    //     use_single_objective_directions = false;
    //     break;
    
    default:
        // Default case is a passthrough.
        approach_mode = approach_kind;
        
        if (approach_kind > 6)
        {
            std::cerr << "Unknown approach " << approach_kind << "." << std::endl;
            std::exit(1);
        }
        break;
    }
}

/**
 * Parses only the options from the command line.
 */
void parseOptions( int argc, char **argv, int *index )
{
    double dummy;

    print_verbose_overview        = 0;
    use_vtr                       = 0;
    use_print_progress_to_screen  = 0;

    use_pre_mutation              = 0;
    use_pre_adaptive_mutation     = 0;
    use_repair_mechanism          = 0;
    stop_population_when_front_is_covered = 0;

    for( ; (*index) < argc; (*index)++ )
    {
        if( argv[*index][0] == '-' )
        {
            /* If it is a negative number, the option part is over */
            if( sscanf( argv[*index], "%lf", &dummy ) && argv[*index][1] != '\0' )
                break;

            if( argv[*index][1] == '\0' )
                optionError( argv, *index );
            // else if( argv[*index][2] != '\0' )
            //     optionError( argv, *index );
            else
            {
                switch( argv[*index][1] )
                {
                    case '?': printUsage(); break;
                    case 'P': printAllInstalledProblems(); break;
                    case 'v': print_verbose_overview        = 1; break;
                    case 'p': use_print_progress_to_screen  = 1; break;
                    case 'm': use_pre_mutation              = 1; break;
                    case 'M': use_pre_adaptive_mutation     = 1; break;
                    case 'r': use_repair_mechanism          = 1; break; 
                    case 'z': stop_population_when_front_is_covered = 1; break;
                    case 's':
                        // Try without c first, then with.
                        if (sscanf(&argv[*index][2], "%i_%i", &arg_population_size, &arg_num_clusters) == 1 )
                        {
                            sscanf(&argv[*index][2], "%i_%c%i", &arg_population_size, &arg_num_cluster_mode, &arg_num_clusters);
                            // std::cout << "Clustering mode kind is " << arg_num_cluster_mode << "." << std::endl;
                        }
                        break;
                    case 'a': 
                        sscanf(&argv[*index][2], "%i_%i_%lf", &approach_kind, &extreme_kernel_mode, &extreme_radius);
                        processApproachMode();
                        break;
                    case 'o': 
                        outpath = &argv[*index][2]; 
                        // std::cout << "Changed path to " << outpath << "!" << std::endl;
                        std::filesystem::create_directories(outpath);
                        break;
                    case 'i':
                        instance = &argv[*index][2];
                        break;
                    case 'f':
                        overrideFrontPath = &argv[*index][2];
                        loadOverrideFront();
                        break;
                    case 'l':
                    {
                        int a = multipass;    
                        sscanf(&argv[*index][2], "%i_%d", &initialization_ls_mode, &a);
                        multipass = a;
                    }
                        break;
                    case 'c':
                        sscanf(&argv[*index][2], "%i_%i", &clustering_mode, &clustering_mode_modifier);
                        break;
                    case 'x':
                        sscanf(&argv[*index][2], "%i_%i", &mixing_pool_mode, &mixing_pool_knn_k_mode);
                        break;
                    case 'q':
                        sscanf(&argv[*index][2], "%lf", &adjusted_nadir_adjustment);
                        break;
                    case 'd': 
                        if (argv[*index][2] == '\0')
                            use_donor_search = true;
                        else
                        {
                            int use_donor_search_int = use_donor_search;
                            sscanf(&argv[*index][2], "%i", &use_donor_search_int);
                            use_donor_search = use_donor_search_int;
                        }
                        break;
                    case 'F':
                    {
                        int a = fi_do_replacement;
                        int b = fi_use_singleobjective;
                        sscanf(&argv[*index][2], "%i_%i_%i", &fi_mode, &a, &b);
                        fi_do_replacement = a;
                        fi_use_singleobjective = b;
                    }
                        break;
                    case 'N':
                    {
                        if (argv[*index][2] == '\0')
                        {
                            // If no arguments: Use NMI (for backwards compatibility)
                            linkage_mode = 1;
                        }
                        else
                        {
                            // Otherwise: set to the value provided
                            sscanf(&argv[*index][2], "%i", &linkage_mode);
                        }
                    }
                        break;
                    case 'R':
                    {
                        // Random Number generator seed
                        sscanf(&argv[*index][2], "%li", &random_seed);
                    }
                        break;
                    default : optionError( argv, *index );
                }
            }
        }
        else /* Argument is not an option, so option part is over */
            break;
    }

    if (use_pre_mutation == TRUE)
        std::cout << "Pre Mutation is enabled.\n";
    if (use_pre_adaptive_mutation == TRUE)
        std::cout << "Pre Adaptive Mutation is enabled.\n";
}
/**
 * Writes the names of all installed problems to the standard output.
 */
void printAllInstalledProblems( void )
{
    int i, n;
    
    n = numberOfInstalledProblems();
    printf("Installed optimization problems:\n");
    for( i = 0; i < n; i++ )
        printf("%3d: %s\n", i, installedProblemName( i ));

    exit( 0 );
}
/**
 * Informs the user of an illegal option and exits the program.
 */
void optionError( char **argv, int index )
{
    printf("Illegal option: %s\n\n", argv[index]);
    printUsage();
}
/**
 * Parses only the EA parameters from the command line.
 */
void parseParameters( int argc, char **argv, int *index )
{
    int noError;

    if( (argc - *index) != 6 )
    {
        printf("Number of parameters is incorrect, require 6 parameters (you provided %d).\n\n", (argc - *index));
        printUsage();
    }

    noError = 1;
    noError = noError && sscanf( argv[*index+0], "%d", &problem_index );
    noError = noError && sscanf( argv[*index+1], "%d", &number_of_objectives );
    noError = noError && sscanf( argv[*index+2], "%d", &number_of_parameters );
    noError = noError && sscanf( argv[*index+3], "%d", &elitist_archive_size_target );
    noError = noError && sscanf( argv[*index+4], "%ld", &maximum_number_of_evaluations );
    noError = noError && sscanf( argv[*index+5], "%ld", &log_progress_interval );
  
    if( !noError )
    {
        printf("Error parsing parameters.\n\n");
        printUsage();
    }
}
/**
 * Prints usage information and exits the program.
 */
void printUsage( void )
{
    printf("Usage: MO-GOMEA [-?] [-P] [-s] [-w] [-v] [-r] [-g] pro dim eas eva log gen\n");
    printf(" -?: Prints out this usage information.\n");
    printf(" -P: Prints out a list of all installed optimization problems.\n");
    printf(" -p: Prints optimization progress to screen.\n");
    printf(" -v: Enables verbose mode. Prints the settings before starting the run.\n");
    printf(" -r: Enables use of a repair mechanism if the problem is constrained.\n");
    printf(" -m: Enables use of the weak mutation operator.\n");
    printf(" -M: Enables use of the strong mutation operator.\n");
    printf(" -z: Enable checking if smaller (inefficient) populations should be stopped.\n");
    printf("\n");
    printf("  pro: Index of optimization problem to be solved.\n");
    printf("  num: Number of objectives to be optimized.\n");
    printf("  dim: Number of parameters.\n");
    printf("  eas: Elitist archive size target.\n");
    printf("  eva: Maximum number of evaluations allowed.\n");
    printf("  log: Interval (in terms of number of evaluations) at which the elitist archive is recorded for logging purposes.\n");
    exit( 0 );
}
/**
 * Checks whether the selected options are feasible.
 */
void checkOptions( void )
{
    if( elitist_archive_size_target < 1 )
    {
        printf("\n");
        printf("Error: elitist archive size target < 1 (read: %d).", elitist_archive_size_target);
        printf("\n\n");

        exit( 0 );
    }
    if( maximum_number_of_evaluations < 1 )
    {
        printf("\n");
        printf("Error: maximum number of evaluations < 1 (read: %ld). Require maximum number of evaluations >= 1.", maximum_number_of_evaluations);
        printf("\n\n");

        exit( 0 );
    }
    if( installedProblemName( problem_index ) == NULL )
    {
        printf("\n");
        printf("Error: unknown index for problem (read index %d).", problem_index );
        printf("\n\n");

        exit( 0 );
    }
}
/**
 * Prints the settings as read from the command line.
 */
void printVerboseOverview( void )
{
    printf("###################################################\n");
    printf("#\n");
    printf("# Problem                 = %s\n", installedProblemName( problem_index ));
    printf("# Number of objectives    = %d\n", number_of_objectives);
    printf("# Number of parameters    = %d\n", number_of_parameters);
    printf("# Elitist ar. size target = %d\n", elitist_archive_size_target);
    printf("# Maximum numb. of eval.  = %ld\n", maximum_number_of_evaluations);
    printf("# Random seed             = %ld\n", random_seed);
    printf("#\n");
    printf("###################################################\n");
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Problems -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
void evaluateIndividual(char *solution, double *obj, double *con, void* metadata, int objective_index_of_extreme_cluster)
{
    number_of_evaluations++;
    if(population_id != -1)
        array_of_number_of_evaluations_per_population[population_id] += 1;

    switch(problem_index)
    {
        case ZEROMAX_ONEMAX: onemaxProblemEvaluation(solution, obj, con, objective_index_of_extreme_cluster); break;
        case TRAP5: trap5ProblemEvaluation(solution, obj, con, objective_index_of_extreme_cluster); break;
        case LOTZ: lotzProblemEvaluation(solution, obj, con, objective_index_of_extreme_cluster); break;
        case KNAPSACK: knapsackProblemEvaluation(solution, obj, con, objective_index_of_extreme_cluster); break;
        case MAXCUT: maxcutProblemEvaluation(solution, obj, con, objective_index_of_extreme_cluster); break;
        case MAXSAT: maxsatProblemEvaluation(solution, obj, con, objective_index_of_extreme_cluster); break;
        case MAXCUT_VS_ONEMAX: maxcutVsOnemaxProblemEvaluation(solution, obj, con, objective_index_of_extreme_cluster); break;
        case BESTOFTRAPS: bestOfTrapsProblemEvaluation(solution, obj, con, metadata, objective_index_of_extreme_cluster); break;
        case BESTOFTRAPS_VS_ONEMAX: bestOfTrapsVsOneMaxProblemEvaluation(solution, obj, con, metadata, objective_index_of_extreme_cluster); break;
        case BESTOFTRAPS_VS_MAXCUT: bestOfTrapsVsMaxCutProblemEvaluation(solution, obj, con, metadata, objective_index_of_extreme_cluster); break;
        case DISCRETIZED_CONTINOUS_PROBLEM: DCProblemEvaluation(solution, obj, con, objective_index_of_extreme_cluster); break;
        default:
            printf("Cannot evaluate this problem!\n");
            exit(1);
    }

    logElitistArchiveAtSpecificPoints();
}

template<typename T>
void performHillClimber(T&& accept,
    char *parent, double *parent_obj, double parent_con, void *parent_metadata,
    char *result, double *obj,        double *con,       void *metadata,
    bool do_update_elitist_archive, bool multipass, int limit = -1)
{
    char *backup;
    double *backup_obj;
    double backup_con;
    void* backup_metadata;

    // No need to copy from parent to result if they are the same.
    if (parent != result)
        copyFromAToB(
            parent, parent_obj, parent_con, parent_metadata,
            result,        obj,        con, metadata);

    // Create a backup.
    backup     = (char*)   Malloc(number_of_parameters * sizeof(char));
    backup_obj = (double*) Malloc(number_of_objectives * sizeof(double));
    initSolutionMetadata(backup_metadata);

    copyFromAToB(
        result,        obj,        *con, metadata, 
        backup, backup_obj, &backup_con, backup_metadata);

    int stop_at = number_of_parameters - 1;
    int current = 0;
    int passes = 0;

    do {
        char current_dimension_size = 2;
        char original = result[current];
        for (int v = 0; v < current_dimension_size; ++v)
        {
            // Would revert the solution back to its original value.
            // Would never be a new improvement.
            // - either a no-op with a wasted function evaluation)
            // - or reverting a previous change -- which was accepted.
            if (v == original) continue;

            // Change single variable.
            result[current] = v;

            // Evaluate change
            evaluateIndividual(result, obj, con, metadata, NOT_EXTREME_CLUSTER);

            char is_new_undominated_point = FALSE;
            char is_dominated_by_archive = FALSE;
            if (do_update_elitist_archive)
            {
                updateElitistArchiveWithReplacementOfExistedMember(result, obj, *con, metadata, &is_new_undominated_point, &is_dominated_by_archive);
            }

            // Check if the change should be accepted.
            // Arguments:
            // double* backup_obj <original objective values>
            // double backup_con <original constraint values>
            // double* obj <new objective values>
            // double *con <new constraint value>
            // bool is_new_undominated_point
            // bool is_dominated_by_archive
            bool accepted = accept(
                backup, backup_obj, backup_con,
                result,        obj,       *con,
                is_new_undominated_point, is_dominated_by_archive);
            // Accept change or reverse.
            if (accepted)
            {
                copyFromAToBSubset(
                    result,        obj,        *con, metadata, 
                    backup, backup_obj, &backup_con, backup_metadata,
                    &current, 1);
                if (multipass)
                    stop_at = current;
            }
            else
            {
                copyFromAToBSubset(
                    backup, backup_obj, backup_con, backup_metadata,
                    result,       obj,        con, metadata,
                    &current, 1);
            }
        }

        // Go to next, wrapping around.
        int original_current = current;
        current = (current + 1) % number_of_parameters;
        if (current < original_current)
            passes += 1;
    } while (current != stop_at && !checkTerminationCondition() && passes != limit);

    free(backup); free(backup_obj); cleanupSolutionMetadata(backup_metadata);
}


/**
 * Returns the name of an installed problem.
 */
char *installedProblemName( int index )
{
    switch( index )
    {
        case  ZEROMAX_ONEMAX:                return( (char *) "Zeromax - Onemax" );
        case  TRAP5:                         return( (char *) "Deceptive Trap 5 - Inverse Trap 5 - Tight Encoding" );
        case  KNAPSACK:                      return( (char *) "Knapsack - 2 Objectives");
        case  LOTZ:                          return( (char *) "Leading One Trailing Zero (LOTZ)");
        case  MAXCUT:                        return( (char *) "Maxcut - 2 Objectives");
        case  MAXSAT:                        return( (char *) "Maxsat & Distance from attractor - 2 objectives");
        case  MAXCUT_VS_ONEMAX:              return( (char *) "Maxcut vs Onemax");
        case  BESTOFTRAPS:                   return( (char *) "Best of Traps");
        case  BESTOFTRAPS_VS_ONEMAX:         return( (char *) "Best of Traps vs OneMax");
        case  BESTOFTRAPS_VS_MAXCUT:         return( (char *) "Best of Traps vs MaxCut");
        case  DISCRETIZED_CONTINOUS_PROBLEM: return( (char *) "Discretized Continuous Problem");
    }
    return( NULL );
}
/**
 * Returns the number of problems installed.
 */
int numberOfInstalledProblems( void )
{
    static int result = -1;
  
    if( result == -1 )
    {
        result = 0;
        while( installedProblemName( result ) != NULL )
            result++;
    }
  
    return( result );
}

/* Problem: 0 - OneMax vs ZeroMax */
void onemaxLoadProblemData()
{
    int k;
    optimization = (char*)Malloc(number_of_objectives*sizeof(char));
    for(k = 0; k < number_of_objectives; k++)
        optimization[k] = MAXIMIZATION;

    use_vtr = 1;
    vtr = 0;
}

void onemaxProblemEvaluation(char *solution, double *obj_values, double *con_value, int /* objective_index_of_extreme_cluster */ )
{
    int i, number_of_1s, number_of_0s;
    
    *con_value = 0.0;
    number_of_0s = 0;
    number_of_1s = 0;
    
    for(i = 0; i < number_of_parameters; i++)
    {
        if(solution[i] == 0)
            number_of_0s++;
        else if(solution[i] == 1)
            number_of_1s++;
    }

    obj_values[0] = number_of_0s;
    obj_values[1] = number_of_1s;
}

double **getDefaultFrontOnemaxZeromax( int *default_front_size )
{
    int  i;
    static double **result = NULL;
    *default_front_size = ( number_of_parameters + 1 );

    if( result == NULL )
    {
        result = (double **) Malloc( (*default_front_size)*sizeof( double * ) );
        for( i = 0; i < (*default_front_size); i++ )
            result[i] = (double *) Malloc( 2*sizeof( double ) );

        for( i = 0; i < (*default_front_size); i++ )
        {
            result[i][0] = i;                                         // Zeromax
            result[i][1] = number_of_parameters - result[i][0];       // Onemax
        }
    }
    return( result );
}

/* Problem: 1 - Trap-5 vs Trap-5 */
void trap5LoadProblemData()
{
    int k;
    optimization = (char*)Malloc(number_of_objectives*sizeof(char));
    for(k = 0; k < number_of_objectives; k++)
        optimization[k] = MAXIMIZATION;

    use_vtr = 1;
    vtr = 0;
}

double deceptiveTrapKTightEncodingFunctionProblemEvaluation( char *parameters, int k, char is_one )
{
    int    i, j, m, u;
    double result;

    if( number_of_parameters % k != 0 )
    {
        printf("Error in evaluating deceptive trap k: Number of parameters is not a multiple of k.\n");
        exit( 0 );
    }

    m      = number_of_parameters / k;
    result = 0.0;
    for( i = 0; i < m; i++ )
    {
        u = 0;
        for( j = 0; j < k; j++ )
            u += (parameters[i*k+j] == is_one) ? 1 : 0;

        if( u == k )
            result += k;
        else
            result += (k-1-u);
    }

    return result;
}

void trap5ProblemEvaluation(char *solution, double *obj_values, double *con_value, int /* objective_index_of_extreme_cluster */ )
{
    *con_value      = 0.0;
    obj_values[0]   = deceptiveTrapKTightEncodingFunctionProblemEvaluation( solution, 5, TRUE );
    obj_values[1]   = deceptiveTrapKTightEncodingFunctionProblemEvaluation( solution, 5, FALSE );
}

double **getDefaultFrontTrap5InverseTrap5( int *default_front_size )
{
    int  i, number_of_blocks;
    static double **result = NULL;

    number_of_blocks = number_of_parameters / 5;
    *default_front_size = ( number_of_blocks + 1 );

    if( result == NULL )
    {
        result = (double **) Malloc( (*default_front_size)*sizeof( double * ) );
        for( i = 0; i < (*default_front_size); i++ )
            result[i] = (double *) Malloc( 2*sizeof( double ) );

        for( i = 0; i < (*default_front_size); i++ )                    // i = number of all-1 blocks
        {
            result[i][0] = ( 5 * i ) + ( 4 * (number_of_blocks - i) ) ;   // Trap-5
            result[i][1] = ( 5 * (number_of_blocks - i)) + ( 4 * i );     // Inverse Trap-5
        }
    }
    return( result );
}

/* Problem: 3 - Leading Ones vs Trailing Zeroes (LOTZ) */
void lotzLoadProblemData()
{
    int k;
    optimization = (char*)Malloc(number_of_objectives*sizeof(char));
    for(k = 0; k < number_of_objectives; k++)
        optimization[k] = MAXIMIZATION;

    use_vtr = 1;
    vtr = 0;
}

void lotzProblemEvaluation(char *solution, double *obj_values, double *con_value, int /* objective_index_of_extreme_cluster */ )
{
    int i;
    double result;

    *con_value = 0.0;
    result = 0.0;
    for(i = 0; i < number_of_parameters; i++)
    {
        if(solution[i] == 0)
            break;
        result += 1;
    }
    obj_values[0] = result; // Leading Ones

    result = 0.0;
    for(i = number_of_parameters - 1; i >= 0; i--)
    {
        if(solution[i] == 1)
            break;
        result += 1;
    }
    obj_values[1] = result; // Trailing Zeros
}

double **getDefaultFrontLeadingOneTrailingZero( int *default_front_size )
{
    int  i;
    static double **result = NULL;

    *default_front_size = ( number_of_parameters + 1 );

    if( result == NULL )
    {
        result = (double **) Malloc( (*default_front_size)*sizeof( double * ) );
        for( i = 0; i < (*default_front_size); i++ )
            result[i] = (double *) Malloc( 2*sizeof( double ) );

        for( i = 0; i < (*default_front_size); i++ )
        {
            result[i][0] = i;                                         // Leading One
            result[i][1] = number_of_parameters - result[i][0];       // Trailing Zero
        }
    }

    return( result );
}

/* Problem: 2 - Multi-Objective Knapsack */
void knapsackLoadProblemData()
{
    int int_number, i, k;
    FILE *file;
    char string[1000];
    double double_number, ratio, *ratios;

    sprintf(string, "./knapsack/knapsack.%d.%d.txt", number_of_parameters, number_of_objectives);
    file = NULL;
    file = fopen(string, "r");
    if(file == NULL)
    {
        printf("Cannot open file %s!\n", string);
        exit(1);
    }

    fscanf(file, "%d", &int_number);
    fscanf(file, "%d", &int_number);

    capacities      = (double*)Malloc(number_of_objectives*sizeof(double));
    weights         = (double**)Malloc(number_of_objectives*sizeof(double*));
    profits         = (double**)Malloc(number_of_objectives*sizeof(double*));
    for(k = 0; k < number_of_objectives; k++)
    {
        weights[k]  = (double*)Malloc(number_of_parameters*sizeof(double));
        profits[k]  = (double*)Malloc(number_of_parameters*sizeof(double));
    }
    for(k = 0; k < number_of_objectives; k++)
    {
        fscanf(file, "%lf", &double_number);
        capacities[k] = double_number;

        for(i = 0; i < number_of_parameters; i++)
        {
            fscanf(file, "%d", &int_number);
            
            fscanf(file, "%d", &int_number);
            weights[k][i] = int_number;
            fscanf(file, "%d", &int_number);
            profits[k][i] = int_number;
        }
    }
    fclose(file);

    ratio_profit_weight = (double*)Malloc(number_of_parameters*sizeof(double));
    for(i = 0; i < number_of_parameters; i++)
    {
        ratio_profit_weight[i] = profits[0][i] / weights[0][i];
        for(k = 1; k < number_of_objectives; k++)
        {
            ratio = profits[k][i] / weights[k][i];
            if(ratio > ratio_profit_weight[i])
                ratio_profit_weight[i] = ratio;
        }
    }

    item_indices_least_profit_order = mergeSort(ratio_profit_weight, number_of_parameters);
    item_indices_least_profit_order_according_to_objective = (int**)Malloc(number_of_objectives*sizeof(int*));
    ratios = (double*)Malloc(number_of_parameters*sizeof(double));
    for(k = 0; k < number_of_objectives; k++)
    {
        for(i = 0; i < number_of_parameters; i++)
            ratios[i] = profits[k][i] / weights[k][i];
        item_indices_least_profit_order_according_to_objective[k] = mergeSort(ratios, number_of_parameters);
    }

    optimization = (char*)Malloc(number_of_objectives*sizeof(char));
    for(k = 0; k < number_of_objectives; k++)
        optimization[k] = MAXIMIZATION;

    free(ratios);
}

void ezilaitiniKnapsackProblemData()
{
    int k;
    for(k = 0; k < number_of_objectives; k++)
    {
        free(weights[k]);
        free(profits[k]);
    }
    free(weights);
    free(profits);
    free(capacities);
    free(ratio_profit_weight);
    free(item_indices_least_profit_order);

    for(k = 0; k < number_of_objectives; k++)
    {
        free(item_indices_least_profit_order_according_to_objective[k]);
    }
    free(item_indices_least_profit_order_according_to_objective);
}

void knapsackSolutionRepair(char *solution, double *solution_profits, double *solution_weights, double *solution_constraint, int objective_index_of_extreme_cluster)
{
    if(objective_index_of_extreme_cluster == -1)
        knapsackSolutionMultiObjectiveRepair(solution, solution_profits, solution_weights, solution_constraint);
    else
        knapsackSolutionSingleObjectiveRepair(solution, solution_profits, solution_weights, solution_constraint, objective_index_of_extreme_cluster);
}

void knapsackSolutionSingleObjectiveRepair(char *solution, double *solution_profits, double *solution_weights, double *solution_constraint, int objective_index)
{
    int i, j, k;
    char isFeasible;

    for(j = 0; j < number_of_parameters; j++)
    {
        i = item_indices_least_profit_order_according_to_objective[objective_index][j];
        if(solution[i] == 0)
            continue;

        solution[i] = 0;
        isFeasible = TRUE;
        for(k = 0; k < number_of_objectives; k++)
        {
            solution_profits[k] = solution_profits[k] - profits[k][i];
            solution_weights[k] = solution_weights[k] - weights[k][i];
            if(solution_weights[k] > capacities[k])
                isFeasible = FALSE;
        }
        if(isFeasible == TRUE)
            break;
    }

    *solution_constraint = 0.0;
    for(k = 0; k < number_of_objectives; k++)
        if(solution_weights[k] > capacities[k])
            (*solution_constraint) = (*solution_constraint) + (solution_weights[k] - capacities[k]);    
}

void knapsackSolutionMultiObjectiveRepair(char *solution, double *solution_profits, double *solution_weights, double *solution_constraint)
{
    int i, j, k;
    char isFeasible;

    for(j = 0; j < number_of_parameters; j++)
    {
        i = item_indices_least_profit_order[j];
        if(solution[i] == 0)
            continue;

        solution[i] = 0;
        isFeasible = TRUE;
        for(k = 0; k < number_of_objectives; k++)
        {
            solution_profits[k] = solution_profits[k] - profits[k][i];
            solution_weights[k] = solution_weights[k] - weights[k][i];
            if(solution_weights[k] > capacities[k])
                isFeasible = FALSE;
        }
        if(isFeasible == TRUE)
            break;
    }

    *solution_constraint = 0.0;
    for(k = 0; k < number_of_objectives; k++)
        if(solution_weights[k] > capacities[k])
            (*solution_constraint) = (*solution_constraint) + (solution_weights[k] - capacities[k]);
}

void knapsackProblemEvaluation(char *solution, double *obj_values, double *con_value, int objective_index_of_extreme_cluster)
{
    int i, k;
    double *solution_profits, *solution_weights;

    solution_weights = (double*)Malloc(number_of_objectives*sizeof(double));
    solution_profits = (double*)Malloc(number_of_objectives*sizeof(double));
    *con_value = 0.0;
    for(k = 0; k < number_of_objectives; k++)
    {
        solution_profits[k] = 0.0;
        solution_weights[k] = 0.0;
        for(i = 0; i < number_of_parameters; i++)
        {
            solution_profits[k] += ((int)solution[i])*profits[k][i];
            solution_weights[k] += ((int)solution[i])*weights[k][i];
        }
        if(solution_weights[k] > capacities[k])
            (*con_value) = (*con_value) + (solution_weights[k] - capacities[k]);
    }

    if(use_repair_mechanism)
    {
        if( (*con_value) > 0)
            knapsackSolutionRepair(solution, solution_profits, solution_weights, con_value, objective_index_of_extreme_cluster);
    }

    for(k = 0; k < number_of_objectives; k++)
        obj_values[k] = solution_profits[k];

    free(solution_weights);
    free(solution_profits);
}

/* Problem: 4 - Maxcut vs Maxcut (MO-GOMEA-Original) */
void maxcutLoadProblemData()
{
    int i, k;
    char string[1000];
    maxcut_edges = (int ***) Malloc(number_of_objectives * sizeof(int **));
    number_of_maxcut_edges = (int *) Malloc(number_of_objectives * sizeof(int ));
    maxcut_edges_weights = (double **) Malloc(number_of_objectives * sizeof(double *));

    for (i = 0; i < number_of_objectives; i++)
    {
        sprintf(string, "maxcut/maxcut_instance_%d_%d.txt", number_of_parameters, i);
        maxcutReadInstanceFromFile(string, i);
    }

    optimization = (char*)Malloc(number_of_objectives*sizeof(char));
    for(k = 0; k < number_of_objectives; k++)
        optimization[k] = MAXIMIZATION;
}

void ezilaitiniMaxcutProblemData()
{
    int i,j;
    for(i=0;i<number_of_objectives;i++)
    {
        for(j=0;j<number_of_maxcut_edges[i];j++)
            free(maxcut_edges[i][j]);
        free(maxcut_edges[i]);
        free(maxcut_edges_weights[i]);
    }
    free(maxcut_edges);
    free(maxcut_edges_weights);
    free(number_of_maxcut_edges); 
}

void maxcutReadInstanceFromFile(char *filename, int objective_index)
{
    char  c, string[1000], substring[1000];
    int   i, j, k, q, number_of_vertices, number_of_edges;
    FILE *file;

    //file = fopen( "maxcut_instance.txt", "r" );
    file = fopen( filename, "r" );
    if( file == NULL )
    {
        printf("Error in opening file \"maxcut_instance.txt\"");
        exit( 0 );
    }

    c = fgetc( file );
    k = 0;
    while( c != '\n' && c != EOF )
    {
        string[k] = (char) c;
        c      = fgetc( file );
        k++;
    }
    string[k] = '\0';

    q = 0;
    j = 0;
    while( (string[j] != ' ') && (j < k) )
    {
        substring[q] = string[j];
        q++;
        j++;
    }
    substring[q] = '\0';
    j++;

    number_of_vertices = atoi( substring );
    if( number_of_vertices != number_of_parameters )
    {
        printf("Error during reading of maxcut instance:\n");
        printf("  Read number of vertices: %d\n", number_of_vertices);
        printf("  Doesn't match number of parameters on command line: %d\n", number_of_parameters);
        exit( 1 );
    }

    q = 0;
    while( (string[j] != ' ') && (j < k) )
    {
        substring[q] = string[j];
        q++;
        j++;
    }
    substring[q] = '\0';
    j++;

    number_of_edges = atoi( substring );
    number_of_maxcut_edges[objective_index] = number_of_edges;
    maxcut_edges[objective_index] = (int **) Malloc( number_of_edges*sizeof( int * ) );
    for( i = 0; i < number_of_edges; i++ )
        maxcut_edges[objective_index][i] = (int *) Malloc( 2*sizeof( int ) );
    maxcut_edges_weights[objective_index] = (double *) Malloc( number_of_edges*sizeof( double ) );

    i = 0;
    c = fgetc( file );
    k = 0;
    while( c != '\n' && c != EOF )
    {
        string[k] = (char) c;
        c      = fgetc( file );
        k++;
    }
    string[k] = '\0';
    while( k > 0 )
    {
        q = 0;
        j = 0;
        while( (string[j] != ' ') && (j < k) )
        {
            substring[q] = string[j];
            q++;
            j++;
        }
        substring[q] = '\0';
        j++;

        maxcut_edges[objective_index][i][0] = atoi( substring )-1;

        q = 0;
        while( (string[j] != ' ') && (j < k) )
        {
            substring[q] = string[j];
            q++;
            j++;
        }
        substring[q] = '\0';
        j++;

        maxcut_edges[objective_index][i][1] = atoi( substring )-1;

        q = 0;
        while( (string[j] != ' ') && (j < k) )
        {
            substring[q] = string[j];
            q++;
            j++;
        }
        substring[q] = '\0';
        j++;

        maxcut_edges_weights[objective_index][i] = atof( substring );
        i++;

        c = fgetc( file );
        k = 0;
        while( c != '\n' && c != EOF )
        {
            string[k] = (char) c;
            c      = fgetc( file );
            k++;
        }
        string[k] = '\0';
    }

    fclose( file );
}

void maxcutProblemEvaluation( char *solution, double *obj_values, double *con_value, int /* objective_index_of_extreme_cluster */ )
{
    int    i, k;
    double result;

    *con_value = 0;

    for(k = 0; k < number_of_objectives; k++)
    {
        result = 0.0;
        for( i = 0; i < number_of_maxcut_edges[k]; i++ )
        {
            if( solution[maxcut_edges[k][i][0]] != solution[maxcut_edges[k][i][1]] )
                result += maxcut_edges_weights[k][i];
        }

        obj_values[k] = result;
    }
}

double** getDefaultFrontMAXCUT( int *default_front_size )
{
    *default_front_size = 0;
    static double **result = NULL;

    if (result == NULL)
    {
        std::filesystem::path maxcut_dir = "./maxcut/";
        char filename[36];
        sprintf(filename, "maxcut_pareto_front_%i.txt", number_of_parameters);
        std::filesystem::path maxcut_instance_front = maxcut_dir / filename;
        std::ifstream front_file(maxcut_instance_front);

        // std::cout << "Loading front from " << maxcut_instance_front << "." << std::endl;

        // Load number of points in front.
        front_file >> (*default_front_size);

        if (front_file.fail())
        {
            // We don't have a front here.

            *default_front_size = 0;
            return NULL;
        }
        
        // std::cout << "Front contains " << *default_front_size << " points." << std::endl;

        // Allocate an array for each point.
        result = (double **) Malloc( (*default_front_size)*sizeof( double * ) );
        for (int i = 0; i < *default_front_size; ++i)
        {
            result[i] = (double*) Malloc(2 * sizeof(double));
            front_file >> result[i][0] >> result[i][1];

            // std::cout << "P" << i << ": " << result[i][0] << ", " << result[i][1] << "\n";
            
            if (front_file.fail())
            {
                std::cerr << "Malformed pareto front for MAXCUT: file " << maxcut_instance_front << " could not be read at number " << i + 1 << std::endl;
                std::exit(1);
            }
        }

        // std::cout << "Successfully loaded front!" << std::endl;
        
    }

    return result;
}

/* Problem: 5 - MaxSat vs Attractor (like OneMax/Zeromax) */
void maxsatLoadProblemData()
{
    if (!instance.has_value())
    {
        std::cerr << "maxsat requires an instance to be defined." << std::endl;
        exit(1);
    }
    std::filesystem::path instancepath = instance.value();

    assert(number_of_objectives == 2);
    optimization = (char*) Malloc(number_of_objectives*sizeof(char));
    optimization[0] = MAXIMIZATION;
    optimization[1] = MAXIMIZATION;

    std::ifstream file;
    file.open(instancepath);
    std::string line;

    int num_variables = 0;
    int num_clauses = -1;
    int clauses_seen = 0;

    maxsat_instance.clear();
    std::vector<int> clause;
    while (std::getline(file, line))
    {
        if (clauses_seen == num_clauses) break;
        // Comment: skip.
        if (line.find('c') == 0){
            continue;
        } 
        // declaration!
        if (line.find('p') == 0)
        {
            std::istringstream iss(line);
            iss.seekg(5);
            iss >> num_variables >> num_clauses;
            continue;
        }
        if (line.find('%') == 0)
        {
            // VTR seperator, read next line.
            std::getline(file, line);
            std::istringstream iss(line);
            int mxsvtr = 0;
            iss >> mxsvtr;
            // TODO: Enable vtr?
            // vtr = mxsvtr;
            // use_vtr = 1;
            // Format was broken the moment the % was encountered.
            // Entire instance must be read at this point.
            break;
        }
        std::istringstream iss(line);
        int variable;
        while (iss >> variable)
        {
            if (variable == 0) {
                std::vector<int> clause_c(clause);
                maxsat_instance.push_back(clause_c);
                clause.clear();
                break;
            }
            clause.push_back(variable);
            assert(variable <= num_variables);
        }
    }
    assert(number_of_parameters == num_variables);

    std::filesystem::path attractor_path = instancepath.replace_extension("attr");
    
    maxsat_attractor.clear();
    // No attractor provided.
    if (! std::filesystem::exists(attractor_path)){
        std::cerr << "No attractor found" << std::endl;
        // exit(1);
        return;
    }
    std::vector<char> attractor;
    attractor.reserve(number_of_parameters);

    {
        std::ifstream f(attractor_path);
        char sym;
        while (f >> sym)
        {
            if (sym == '0') attractor.push_back(0);
            if (sym == '1') attractor.push_back(1);
            if (sym == '\n')
            {
                maxsat_attractor.push_back(attractor);
                attractor.clear();
            }
        }
    }

    maxsat_attractor.push_back(attractor);

    assert(attractor.size() == (size_t) number_of_parameters);
}

void maxsatProblemEvaluation( char *solution, double *obj_values, double *con_value, int /* objective_index_of_extreme_cluster */ )
{
    *con_value = 0;

    // MAXSAT
    double result = 0.0;
    for (std::vector<int> clause: maxsat_instance) {
        for (int var: clause) {
            if (solution[abs(var) - 1] == (var > 0)) {
                result += 1;
                break;
            }
        }
    }
    obj_values[0] = result - maxsat_instance.size();
    // std::cout << "Evaluated solution with " << result << " clauses satisfied out of " << maxsat_instance.size() << std::endl;

    // Distance from a notable attractor.
    // (more general would be a current best... But...)
    int agg = number_of_parameters + 1;
    if (! maxsat_attractor.empty())
    {
        for (std::vector<char> ma: maxsat_attractor)
        {
            int different = 0;
            for (size_t i = 0; i < (size_t) number_of_parameters; ++i)
            {
                different += (solution[i] != ma[i]);
            }
            agg = std::min(agg, different);
        }
    }
    if (maxsat_attractor.size() > 0)
    {
        obj_values[1] = agg;
    }
    else
    {
        // No attractors. Set to 0.
        obj_values[1] = 0;
    }
}

double** getDefaultFrontMAXSAT( int *default_front_size )
{
    *default_front_size = 0;
    static double **result = NULL;
    if (result == NULL)
    {
        std::cout << "Getting default front";
        if (maxsat_vtr.has_value())
            *default_front_size = 1;
    }

    if (result == NULL)
    {
        result = (double **) Malloc( (*default_front_size)*sizeof( double * ) );
        if (maxsat_vtr.has_value())
        {
            std::cout << "MAXSAT with target " << maxsat_vtr.value() << std::endl;
            result[0] = (double*) Malloc(2 * sizeof(double));
            result[0][0] = maxsat_vtr.value();
            result[0][1] = 0;
        }
    }

    return result;
}

// Problem: 6 - MAXCUT vs OneMax
struct MaxCutInstance
{
    size_t num_vertices;
    size_t num_edges;
    std::vector<std::tuple<int, int, int>> graph;
};

MaxCutInstance load_maxcut(std::filesystem::path instancePath)
{
    MaxCutInstance instance;
    std::ifstream in(instancePath);

    in >> instance.num_vertices >> instance.num_edges;

    instance.graph.resize(instance.num_edges);
    
    for (size_t e = 0; e < instance.num_edges; ++e)
    {
        size_t from;
        size_t to;
        int weight;

        in >> from >> to >> weight;
        instance.graph[e] = std::make_tuple(from - 1, to - 1, weight);
    }

    in.close();
    
    return instance;
}

int evaluate_maxcut(MaxCutInstance &instance, char solution[], size_t /* solution_size */)
{
    // assert(instance.num_vertices == solution_size);
    int weight = 0;
    for (std::tuple<size_t, size_t, int> edge: instance.graph)
    {
        size_t from = std::get<0>(edge);
        size_t to = std::get<1>(edge);
        int w = std::get<2>(edge);

        weight += w * (solution[from] != solution[to]);
    }
    return weight;
}

int evaluate_onemax(char solution[], size_t solution_size)
{
    int count = 0;
    for (size_t i = 0; i < solution_size; ++i)
    {
        count += solution[i];
    }
    return count;
}


MaxCutInstance global_instance;

void maxcutVsOnemaxLoadProblem()
{
    assert(instance.has_value());
    std::filesystem::path instancepath = instance.value();
    global_instance = load_maxcut(instancepath);
    assert(number_of_parameters == static_cast<int>(global_instance.num_vertices));

    assert(number_of_objectives == 2);
    optimization = (char*)Malloc(number_of_objectives * sizeof(char));
    optimization[0] = MAXIMIZATION;
    optimization[1] = MAXIMIZATION;
}

void maxcutVsOnemaxProblemEvaluation( char *solution, double *obj_values, double *con_value, int /* objective_index_of_extreme_cluster */ )
{
    *con_value = 0;
    obj_values[0] = evaluate_maxcut(global_instance, solution, number_of_parameters);
    obj_values[1] = evaluate_onemax(solution, number_of_parameters);
}

/* Problem: 7 - Best-of-Traps vs Best-of-Traps */
struct PermutedRandomTrap
{
    int number_of_parameters;
    int block_size;
    std::vector<size_t> permutation;
    std::vector<char> optimum;
};

int trapFunction(int unitation, int size)
{
    if (unitation == size) return size;
    return size - unitation - 1;
}

int evaluateConcatenatedPermutedTrap( PermutedRandomTrap &permutedRandomTrap, char* solution )
{
    int l = permutedRandomTrap.number_of_parameters;
    // int number_of_blocks = l / block_size;
    
    int objective = 0;
    for (int block_start = 0; block_start < l; block_start += permutedRandomTrap.block_size)
    {
        int unitation = 0;
        int current_block_size = std::min(permutedRandomTrap.block_size, l - block_start);
        for (int i = 0; i < current_block_size; ++i)
        {
            int idx = permutedRandomTrap.permutation[block_start + i];
            unitation += solution[idx] == permutedRandomTrap.optimum[idx];
        }
        objective += trapFunction(unitation, current_block_size);
    }
    return objective;
}

PermutedRandomTrap generatePermutedRandomTrap(std::mt19937 &rng, int n, int k)
{
    // Generate permutation
    std::vector<size_t> permutation(n);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::shuffle(permutation.begin(), permutation.end(), rng);
    // Generate optimum
    std::vector<char> optimum(n);
    std::uniform_int_distribution<char> binary_dist(0, 1);
    std::generate(optimum.begin(), optimum.end(), [&rng, &binary_dist](){return binary_dist(rng);});

    return PermutedRandomTrap {
        /* .number_of_parameters = */ n,
        /* .block_size = */ k,
        /* .permutation = */ permutation,
        /* .optimimum = */ optimum
    };
}

struct BestOfTraps
{
    std::vector<PermutedRandomTrap> permutedRandomTraps;
};

int evaluateBestOfTraps( BestOfTraps &bestOfTraps, char* solution, int &best_fn )
{
    int result = INT_MIN;
    for (size_t fn = 0; fn < bestOfTraps.permutedRandomTraps.size(); ++fn)
    {
        int result_subfn = evaluateConcatenatedPermutedTrap(bestOfTraps.permutedRandomTraps[fn], solution);
        if (result_subfn > result)
        {
            best_fn = fn;
            result = result_subfn;
        }
    }
    return result;
}

BestOfTraps generateBestOfTrapsInstance(int64_t seed, int n, int k, int fns)
{
    std::vector<PermutedRandomTrap> randomPermutedTraps(fns);
    std::mt19937 rng(seed);

    for (int fn = 0; fn < fns; ++fn)
    {
        randomPermutedTraps[fn] = generatePermutedRandomTrap(rng, n, k);
    }
    
    return BestOfTraps {
        randomPermutedTraps
    };
}

void writeBestOfTraps(std::filesystem::path outpath, BestOfTraps &bot)
{
    std::ofstream file(outpath);
    file << bot.permutedRandomTraps.size() << '\n';
    for (PermutedRandomTrap subfunction: bot.permutedRandomTraps)
    {
        file << subfunction.number_of_parameters << ' ';
        file << subfunction.block_size << '\n';
        // optimum
        for (int o = 0; o < subfunction.number_of_parameters; ++o)
        {
            file << static_cast<int>(subfunction.optimum[o]);
            if (o == subfunction.number_of_parameters - 1)
                file << '\n';
            else
                file << ' ';
        }
        // permutation
        for (int o = 0; o < subfunction.number_of_parameters; ++o)
        {
            file << subfunction.permutation[o];
            if (o == subfunction.number_of_parameters - 1)
                file << '\n';
            else
                file << ' ';
        }
    }
}

void stopInvalidInstanceBOT(std::ifstream &stream, std::string expected)
{
    std::cerr << "Instance provided for BOT is invalid.\n";
    std::cerr << "Invalid character at position " << stream.tellg() << ".\n";
    std::cerr << expected << std::endl;
    exit(1);
}
void stopFileMissingBOT(std::filesystem::path file)
{
    std::cerr << "Instance provided for BOT is invalid.\n";
    std::cerr << "File " << file << " does not exist." << std::endl;
    exit(1);
}


BestOfTraps readBestOfTraps(std::filesystem::path inpath)
{
    if (! std::filesystem::exists(inpath)) stopFileMissingBOT(inpath);
    std::ifstream file(inpath);
    size_t number_of_subfunctions = 0;
    file >> number_of_subfunctions;
    if (file.fail()) stopInvalidInstanceBOT(file, "expected number_of_subfunctions"); 

    std::vector<PermutedRandomTrap> subfunctions;
    subfunctions.reserve(number_of_subfunctions);

    for (int fn = 0; fn < static_cast<int>(number_of_subfunctions); ++fn)
    {
        int number_of_parameters = -1;
        int block_size = -1;
        file >> number_of_parameters;
        if (file.fail()) stopInvalidInstanceBOT(file, "expected number_of_parameters"); 
        file >> block_size;
        if (file.fail()) stopInvalidInstanceBOT(file, "expected block_size");
        std::string current_line;
        // Skip to the next line.
        if(! std::getline(file, current_line)) stopInvalidInstanceBOT(file, "expected newline");
        // optimum
        std::vector<char> optimum;
        optimum.reserve(number_of_parameters);
        if(! std::getline(file, current_line)) stopInvalidInstanceBOT(file, "expected optimum");
        
        {
            std::stringstream linestream(current_line);
            int v = 0;
            while (!linestream.fail())
            {
                linestream >> v;
                optimum.push_back(static_cast<char>(v));
            }
        }
        std::vector<size_t> permutation;
        permutation.reserve(number_of_parameters);
        if(! std::getline(file, current_line)) stopInvalidInstanceBOT(file, "expected permutation");
        {
            std::stringstream linestream(current_line);
            int v = 0;
            while (!linestream.fail())
            {
                linestream >> v;
                permutation.push_back(v);
            }
        }
        
        PermutedRandomTrap prt = PermutedRandomTrap {
            number_of_parameters,
            block_size,
            permutation,
            optimum
        };

        subfunctions.push_back(prt);
    }
    return BestOfTraps {
        subfunctions
    };
}

void stopInvalidInstanceSpecifierBOT(std::stringstream &stream, std::string expected)
{
    int fail_pos = stream.tellg();
    std::string s;
    stream >> s;
    std::cerr << "While loading Best of Traps instance string was invalid at position " << fail_pos << ".\n";
    std::cerr << expected << ". Remainder is `" << s << "`.\n";
    std::cerr << "Expected format is `g(e?)_<n>_<k>_<fns>_<seed>` for generating a bot instance (include e to export to file).\n";
    std::cerr << "and `l_<path>` for loading from a file.\n";
    std::cerr << "Each instance is their own dimension, and each instance/dimension is separated by a `;`" << std::endl; //  or `f_<path>`
    exit(1);
}

std::vector<BestOfTraps> bestOfTrapsInstances;

void loadBestOfTraps()
{
    if (! instance.has_value())
    {
        std::cerr << "While loading Best of Traps did not provide instance (via -i)\n";
        std::cerr << "Expected format: `g_<n>_<k>_<fns>_<seed>` for each dimension, with dimensions separated by `;`"; //  or `f_<path>`
        exit(1);
    }

    std::stringstream instance_stream(instance.value());

    bestOfTrapsInstances.clear();
    while (! instance_stream.eof())
    {
        std::string t;
        if(! std::getline(instance_stream, t, '_')) stopInvalidInstanceSpecifierBOT(instance_stream, "expected <str>_\n");

        if (t[0] == 'g') // Generate
        {
            bool write_to_file = t.length() >= 2 && t[1] == 'e';
            
            // Parameters, initialized to silence warnings.
            int n = -1, k = -1, fns = -1; size_t seed = 0U;
            instance_stream >> n;
            if(instance_stream.fail()) stopInvalidInstanceSpecifierBOT(instance_stream, "expected integer");
            if(instance_stream.get() != '_') stopInvalidInstanceSpecifierBOT(instance_stream, "expected `_`");
            instance_stream >> k;
            if(instance_stream.fail()) stopInvalidInstanceSpecifierBOT(instance_stream, "expected integer");
            if(instance_stream.get() != '_') stopInvalidInstanceSpecifierBOT(instance_stream, "expected `_`");
            instance_stream >> fns;
            if(instance_stream.fail()) stopInvalidInstanceSpecifierBOT(instance_stream, "expected integer");
            if(instance_stream.get() != '_') stopInvalidInstanceSpecifierBOT(instance_stream, "expected `_`");
            instance_stream >> seed;
            if(instance_stream.fail()) stopInvalidInstanceSpecifierBOT(instance_stream, "expected integer");

            BestOfTraps bot = generateBestOfTrapsInstance(seed, n, k, fns);
            if (write_to_file)
            {
                std::filesystem::path botoutdirectory = "./bestoftraps/";
                if (!std::filesystem::exists(botoutdirectory))
                {
                    std::filesystem::create_directories(botoutdirectory);
                }
                std::string filename = "bot_n" + std::to_string(n) + "k" + std::to_string(k) + "fns" + std::to_string(fns) + "s"  + std::to_string(seed) + ".txt";
                std::filesystem::path botoutpath = botoutdirectory / filename;
                writeBestOfTraps(botoutpath, bot);
            }
            bestOfTrapsInstances.push_back(bot);

            if(instance_stream.get() != ';') break;
        }
        else if (t[0] == 'f')
        {
            std::string botinpathstr;
            if(! std::getline(instance_stream, botinpathstr, ';')) stopInvalidInstanceSpecifierBOT(instance_stream, "expected path");
            std::filesystem::path botinpath = botinpathstr;
            BestOfTraps bot = readBestOfTraps(botinpath);
            bestOfTrapsInstances.push_back(bot);
        }
        else stopInvalidInstanceSpecifierBOT(instance_stream, "expected one of {`g`, `f`}");
    }

    // Ensure all instances have the right size. (i.e. not larger than the number of parameters)
    for (BestOfTraps instance: bestOfTrapsInstances)
    {
        for (PermutedRandomTrap subfunction: instance.permutedRandomTraps)
        {
            assert(number_of_parameters >= subfunction.number_of_parameters);
        }
    }
    // Ensure the number of objectives is correct.
    assert(number_of_objectives == static_cast<int>(bestOfTrapsInstances.size()));

    // Set optimization direction
    optimization = (char*)Malloc(number_of_objectives * sizeof(char));
    for (int o = 0; o < number_of_objectives; ++o)
    {
        optimization[o] = MAXIMIZATION;
    }
}

struct BestOfTrapsSolutionMetadata
{
    std::vector<int> active_trap_per_dimension;
};


BestOfTrapsSolutionMetadata* getBestOfTrapsMetadata(void* &metadata)
{
    
    return (BestOfTrapsSolutionMetadata*) metadata;
}

BestOfTrapsSolutionMetadata* initBestOfTrapsMetadata()
{
    if (use_metadata)
    {
        return new BestOfTrapsSolutionMetadata {
            std::vector<int>()
        };
    }
    return NULL;
}

void cleanupBestOfTrapsMetadata(void* &metadata)
{
    BestOfTrapsSolutionMetadata* bot_metadata = getBestOfTrapsMetadata(metadata);
    if (use_metadata)
        delete bot_metadata;
    // Otherwise metadata should be null.
}

void copyBestOfTrapsSolutionMetadata(void* &metadata_from, void* &metadata_to)
{
    if (use_metadata)
    {
        // volatile int a = 0;
        if (metadata_from == NULL || metadata_to == NULL)
        {
            std::cerr << "Got null metadata while metadata should be non-null for problem index.\n";
            std::cerr << "This is a bug that should be fixed: somehow memory was not initialized.\n";
            std::cerr << "metadata_from: " << metadata_from << "; metadata_to: " << metadata_to << "\n";
            // std::cerr << "Stacktrace:\n" << boost::stacktrace::stacktrace() << "\n";
        }

        assert(metadata_from != NULL);
        assert(metadata_to != NULL);
        BestOfTrapsSolutionMetadata *bot_metadata_from = getBestOfTrapsMetadata(metadata_from);
        BestOfTrapsSolutionMetadata *bot_metadata_to = getBestOfTrapsMetadata(metadata_to);
        bot_metadata_to->active_trap_per_dimension = bot_metadata_from->active_trap_per_dimension;
    }
    else
    {
        assert(metadata_from == NULL);
        assert(metadata_to == NULL);
    }
}

// void writeCSVHeadersMetadata(std::ostream &outstream);
// void writeCSVMetadata(std::ostream &outstream, void* metadata);
bool writeCSVHeaderBestOfTrapsMetadata(std::ostream &outstream, bool has_preceding_field)
{
    if (use_metadata)
    {
        if (has_preceding_field)
            outstream << ',';
        outstream << "best_traps";
        return true;
    }
    return false;
}

bool writeCSVBestOfTrapsMetadata(std::ostream &outstream, void* metadata, bool has_preceding_field)
{
    if (use_metadata)
    {
        BestOfTrapsSolutionMetadata *bot_metadata = getBestOfTrapsMetadata(metadata);
        if (has_preceding_field)
            outstream << ',';
        outstream << '"';
        size_t num_dims = bot_metadata->active_trap_per_dimension.size();
        for (size_t d = 0; d < num_dims; ++d)
        {
            int best_trap_id = bot_metadata->active_trap_per_dimension[d];
            outstream << best_trap_id;
            if (d != num_dims - 1)
                outstream << ",";
        }
        outstream << '"';
        return true;
    }
    return false;
}

void bestOfTrapsProblemEvaluation( char *solution, double *obj_values, double *con_value, void* metadata, int /* objective_index_of_extreme_cluster */ )
{
    BestOfTrapsSolutionMetadata *bot_metadata = getBestOfTrapsMetadata(metadata);
    *con_value = 0;
    
    if (use_metadata)
    {
        bot_metadata->active_trap_per_dimension.resize(bestOfTrapsInstances.size());
    }
    
    for (size_t o = 0; o < bestOfTrapsInstances.size(); ++o)
    {
        int best_subfn = 0;
        obj_values[o] = evaluateBestOfTraps(bestOfTrapsInstances[o], solution, best_subfn);
        
        if (use_metadata)
        {
            bot_metadata->active_trap_per_dimension[o] = best_subfn;
        }
    }
}

double** getDefaultFrontBestOfTraps( int *default_front_size )
{
    *default_front_size = 0;
    static double **result = NULL;

    if (result == NULL)
    {
        // Standard reference front for best of traps contains the single-objective
        // optima. This means normalized HV can be (significantly!) greater than 1.
        // Providing an override reference front is recommended!
        *default_front_size = static_cast<int>(bestOfTrapsInstances.size());
        result = (double **) Malloc( (*default_front_size)*sizeof( double * ) );
        
        for (int d = 0; d < *default_front_size; ++d)
        {
            result[d] = (double*) Malloc(number_of_objectives * sizeof(double));
            // Initialize to 0.
            for (int o = 0; o < number_of_objectives; ++o)
            {
                result[d][o] = 0;
            }
            int max_n = 0;
            for (PermutedRandomTrap subfunction: bestOfTrapsInstances[d].permutedRandomTraps)
            {
                max_n = std::max(max_n, subfunction.number_of_parameters);
            }
            result[d][d] = max_n;
        }
    }

    return result;
}

/* Problem 8: BOT vs OneMax */
void loadBestOfTrapsVsOneMax()
{
    /* Start: Copied from loadBestOfTraps() */
    if (! instance.has_value())
    {
        std::cerr << "While loading Best of Traps vs OneMax did not provide instance (via -i)\n";
        std::cerr << "Expected format: `g_<n>_<k>_<fns>_<seed>` for each dimension, with dimensions separated by `;`"; //  or `f_<path>`
        exit(1);
    }

    std::stringstream instance_stream(instance.value());

    bestOfTrapsInstances.clear();
    while (! instance_stream.eof())
    {
        std::string t;
        if(! std::getline(instance_stream, t, '_')) stopInvalidInstanceSpecifierBOT(instance_stream, "expected <str>_\n");

        if (t[0] == 'g') // Generate
        {
            bool write_to_file = t.length() >= 2 && t[1] == 'e';
            
            // Parameters, initialized to silence warnings.
            int n = -1, k = -1, fns = -1; size_t seed = 0U;
            instance_stream >> n;
            if(instance_stream.fail()) stopInvalidInstanceSpecifierBOT(instance_stream, "expected integer");
            if(instance_stream.get() != '_') stopInvalidInstanceSpecifierBOT(instance_stream, "expected `_`");
            instance_stream >> k;
            if(instance_stream.fail()) stopInvalidInstanceSpecifierBOT(instance_stream, "expected integer");
            if(instance_stream.get() != '_') stopInvalidInstanceSpecifierBOT(instance_stream, "expected `_`");
            instance_stream >> fns;
            if(instance_stream.fail()) stopInvalidInstanceSpecifierBOT(instance_stream, "expected integer");
            if(instance_stream.get() != '_') stopInvalidInstanceSpecifierBOT(instance_stream, "expected `_`");
            instance_stream >> seed;
            if(instance_stream.fail()) stopInvalidInstanceSpecifierBOT(instance_stream, "expected integer");

            BestOfTraps bot = generateBestOfTrapsInstance(seed, n, k, fns);
            if (write_to_file)
            {
                std::filesystem::path botoutdirectory = "./bestoftraps/";
                if (!std::filesystem::exists(botoutdirectory))
                {
                    std::filesystem::create_directories(botoutdirectory);
                }
                std::string filename = "bot_n" + std::to_string(n) + "k" + std::to_string(k) + "fns" + std::to_string(fns) + "s"  + std::to_string(seed) + ".txt";
                std::filesystem::path botoutpath = botoutdirectory / filename;
                writeBestOfTraps(botoutpath, bot);
            }
            bestOfTrapsInstances.push_back(bot);

            if(instance_stream.get() != ';') break;
        }
        else if (t[0] == 'f')
        {
            std::string botinpathstr;
            if(! std::getline(instance_stream, botinpathstr, ';')) stopInvalidInstanceSpecifierBOT(instance_stream, "expected path");
            std::filesystem::path botinpath = botinpathstr;
            BestOfTraps bot = readBestOfTraps(botinpath);
            bestOfTrapsInstances.push_back(bot);
        }
        else stopInvalidInstanceSpecifierBOT(instance_stream, "expected one of {`g`, `f`}");
    }
    
    // Ensure all instances have the right size. (i.e. not larger than the number of parameters)
    for (BestOfTraps instance: bestOfTrapsInstances)
    {
        for (PermutedRandomTrap subfunction: instance.permutedRandomTraps)
        {
            assert(number_of_parameters >= subfunction.number_of_parameters);
        }
    }

    // Unlike the plain BestOfTraps, there is an extra objective in the form of OneMax
    assert(number_of_objectives == static_cast<int>(bestOfTrapsInstances.size()) + 1);

    // Set optimization direction
    optimization = (char*)Malloc(number_of_objectives * sizeof(char));
    for (int o = 0; o < number_of_objectives; ++o)
    {
        optimization[o] = MAXIMIZATION;
    }
}

void bestOfTrapsVsOneMaxProblemEvaluation( char *solution, double *obj_values, double *con_value, void* metadata, int /* objective_index_of_extreme_cluster */ )
{
    BestOfTrapsSolutionMetadata *bot_metadata = getBestOfTrapsMetadata(metadata);

    if (use_metadata)
    {
        bot_metadata->active_trap_per_dimension.resize(bestOfTrapsInstances.size());
    }

    *con_value = 0;
    for (size_t o = 0; o < bestOfTrapsInstances.size(); ++o)
    {
        int best_subfn = 0;
        obj_values[o] = evaluateBestOfTraps(bestOfTrapsInstances[o], solution, best_subfn);
        if (use_metadata)
        {
            bot_metadata->active_trap_per_dimension[o] = best_subfn;
        }
    }
    obj_values[bestOfTrapsInstances.size()] = evaluate_onemax(solution, number_of_parameters);
}

/* Problem: 9 - Best-of-Traps vs MaxCut */
void stopInvalidInstanceSpecifierBOTMaxCut(std::stringstream &stream, std::string expected)
{
    int fail_pos = stream.tellg();
    std::string s;
    stream >> s;
    std::cerr << "While loading Best of Traps or MaxCut instance string was invalid at position " << fail_pos << ".\n";
    std::cerr << expected << ". Remainder is `" << s << "`.\n";
    std::cerr << "Expected format is  `b_<path>` (for BOT) or `c_<path>` (for MaxCut) for each dimension.\n";
    std::cerr << "Each instance is their own dimension, and each instance/dimension is separated by a `;`" << std::endl; //  or `f_<path>`
    exit(1);
}


std::vector<std::variant<BestOfTraps, MaxCutInstance>> bestOfTrapsVsMaxCutInstances;
void loadBestOfTrapsVsMaxCut()
{
    /* Start: Copied from loadBestOfTraps() */
    if (! instance.has_value())
    {
        std::cerr << "While loading Best of Traps vs MaxCut did not provide instance (via -i)\n";
        std::cerr << "Expected format: `b_<path>` (for BOT) or `c_<path>` (for MaxCut) for each dimension, with dimensions separated by `;`"; //  or `f_<path>`
        exit(1);
    }

    std::stringstream instance_stream(instance.value());

    bestOfTrapsVsMaxCutInstances.clear();
    while (! instance_stream.eof())
    {
        std::string t;
        if(! std::getline(instance_stream, t, '_')) stopInvalidInstanceSpecifierBOTMaxCut(instance_stream, "expected <str>_\n");

        if (t[0] == 'b') // Best of Traps
        {
            std::string botinpathstr;
            if(! std::getline(instance_stream, botinpathstr, ';')) stopInvalidInstanceSpecifierBOTMaxCut(instance_stream, "expected path");
            std::filesystem::path botinpath = botinpathstr;
            BestOfTraps bot = readBestOfTraps(botinpath);
            bestOfTrapsVsMaxCutInstances.push_back(bot);
        }
        else if (t[0] == 'c') // MaxCut
        {
            std::string maxcutinpathstr;
            if(! std::getline(instance_stream, maxcutinpathstr, ';')) stopInvalidInstanceSpecifierBOTMaxCut(instance_stream, "expected path");
            std::filesystem::path botinpath = maxcutinpathstr;
            MaxCutInstance maxcut = load_maxcut(botinpath);
            bestOfTrapsVsMaxCutInstances.push_back(maxcut);
        }
        else stopInvalidInstanceSpecifierBOTMaxCut(instance_stream, "expected one of {`g`, `f`}");
    }
    
    // Ensure all instances have the right size. (i.e. not larger than the number of parameters)
    for (std::variant<BestOfTraps, MaxCutInstance> instance: bestOfTrapsVsMaxCutInstances)
    {
        if (std::holds_alternative<BestOfTraps>(instance))
        {
            BestOfTraps &botinstance = std::get<BestOfTraps>(instance);
            for (PermutedRandomTrap subfunction: botinstance.permutedRandomTraps)
            {
                assert(number_of_parameters >= subfunction.number_of_parameters);
            }
        }
        else if (std::holds_alternative<MaxCutInstance>(instance))
        {
            MaxCutInstance &maxcutinstance = std::get<MaxCutInstance>(instance);
            assert(number_of_parameters >= static_cast<int>(maxcutinstance.num_vertices));
        }// else is impossible.
    }

    assert(number_of_objectives == static_cast<int>(bestOfTrapsVsMaxCutInstances.size()));

    // Set optimization direction
    optimization = (char*)Malloc(number_of_objectives * sizeof(char));
    for (int o = 0; o < number_of_objectives; ++o)
    {
        optimization[o] = MAXIMIZATION;
    }
}

void bestOfTrapsVsMaxCutProblemEvaluation( char *solution, double *obj_values, double *con_value, void *metadata, int /* objective_index_of_extreme_cluster */ )
{
    BestOfTrapsSolutionMetadata *bot_metadata = getBestOfTrapsMetadata(metadata);
    *con_value = 0;

    if (use_metadata)
    {
        bot_metadata->active_trap_per_dimension.resize(bestOfTrapsVsMaxCutInstances.size());
    }

    for (size_t o = 0; o < bestOfTrapsVsMaxCutInstances.size(); ++o)
    {
        auto instance = bestOfTrapsVsMaxCutInstances[o];
        if (std::holds_alternative<BestOfTraps>(instance))
        {
            int best_subfn = 0;
            obj_values[o] = evaluateBestOfTraps(std::get<BestOfTraps>(instance), solution, best_subfn);
            if (use_metadata)
            {
                bot_metadata->active_trap_per_dimension[o] = best_subfn; 
            }
        }
        else if (std::holds_alternative<MaxCutInstance>(instance))
        {
            obj_values[o] = evaluate_maxcut(std::get<MaxCutInstance>(instance), solution, number_of_parameters);
            if (use_metadata)
            {
                bot_metadata->active_trap_per_dimension[o] = -1;
            }
        }
    }
}

/* Problem 10: Discretized ZDT */

// Define evaluation functions for their continuous counterparts.

void evaluateContinuousZDT1(int m, double* x, double* obj_values)
{
    // x in [0, 1]

    double f1 = x[0];
    double g = 0;
    for (int i = 1; i < m; ++i)
    {
        g += x[i];
    }
    g = 1 + 9 * g / (m - 1);
    double h = 1 - std::sqrt(f1 / g);

    // Negated as to provide a maximization problem
    obj_values[0] = - f1;
    obj_values[1] = - g * h;
}

void evaluateContinuousZDT2(int m, double* x, double* obj_values)
{
    double f1 = x[0];
    double g = 0;
    for (int i = 1; i < m; ++i)
    {
        g += x[i];
    }
    g = 1 + 9 * g / (m - 1);
    double h = 1 - (f1 / g) * (f1 / g);

    // Negated as to provide a maximization problem
    obj_values[0] = - f1;
    obj_values[1] = - g * h;
}

void evaluateContinuousZDT3(int m, double* x, double* obj_values)
{
    double f1 = x[0];
    double g = 0;
    for (int i = 1; i < m; ++i)
    {
        g += x[i];
    }
    g = 1 + 9 * g / (m - 1);
    double h = 1 - std::sqrt(f1 / g) - (f1 / g) * std::sin(10.0 * M_PI * f1);

    // Negated as to provide a maximization problem
    obj_values[0] = - f1;
    obj_values[1] = - g * h;
}

void evaluateContinuousZDT4(int m, double* x, double* obj_values)
{
    double f1 = x[0];
    double g = 0;
    for (int i = 1; i < m; ++i)
    {
        // Scale x[i] such that [0, 1] -> [-5, 5]
        double x_i = x[i] * 10 - 5;
        g += x_i * x_i - 10.0 * std::cos(4 * M_PI * x_i);
    }
    g = 1 + 10 * (m - 1) + g;
    double h = 1 - std::sqrt(f1 / g) - (f1 / g) * std::sin(10.0 * M_PI * f1);

    // Negated as to provide a maximization problem
    obj_values[0] = - f1;
    obj_values[1] = - g * h;
}

// Lookup table for various continuous functions.
const std::map<std::string, std::function<void(int, double*, double*)>> dcp_continuous_functions = {
    {"zdt1", evaluateContinuousZDT1},
    {"ZDT1", evaluateContinuousZDT1},
    {"zdt2", evaluateContinuousZDT2},
    {"ZDT2", evaluateContinuousZDT2},
    {"zdt3", evaluateContinuousZDT3},
    {"ZDT3", evaluateContinuousZDT3},
    {"zdt4", evaluateContinuousZDT4},
    {"ZDT4", evaluateContinuousZDT4},
};

double decodeBinarySpaced01(char* bits, int num_bits)
{
    int v = 1 << (num_bits - 1);
    int r = 0;
    // Pack bits into an integer. 
    for (int i = num_bits - 1; i >= 0; --i)
    {
        r |= bits[i] * v;
        v = v >> 1;
    }
    return static_cast<double>(r) / (static_cast<double>((1 << (num_bits - 1)) - 1));
}

double decodeGraySpaced01(char* bits, int num_bits)
{
    int v = 1 << (num_bits - 1);
    char parity = 0;
    int r = 0;
    // Pack bits into an integer, while xor-ing them with the more significant bits.
    // Array is reversed to keep track the current parity efficiently.
    for (int i = num_bits - 1; i >= 0; --i)
    {
        parity = bits[i] ^ parity;
        r |= parity * v;
        v = v >> 1;
    }
    return static_cast<double>(r) / (static_cast<double>(1 << ((num_bits) - 1)));
}

struct DiscretizedContinousProblem
{
    // Continuous evaluation function.
    // Arguments are (number_of_continuous_variables, continuous_parameter_values, objective_values)
    std::function<void(int, double*, double*)> function;
    // Number of continuous variables.
    int m;
    // How many bits to use per continuous variable.
    int bits_per_variable;
    // 0 - binary code indexing equally spaced points from [0, 1]
    // 1 - gray code indexing equally spaced points from [0, 1]
    int discretizer;
};

void invalidDCPInstanceSpecifier()
{
    std::cerr << "Instance specifier is invalid for DCP.\n";
    std::cerr << "Format: <function name>_<bits per variable>_<number of continuous variables>_<discretizer>.\n";
    std::exit(1);
}

DiscretizedContinousProblem loadDiscretizedContinuousProblem(std::string &instance)
{
    std::stringstream s_instance(instance);

    std::string function_name;
    int m = 0;
    int bits_per_variable;
    int discretizer;

    if (! std::getline(s_instance, function_name, '_')) invalidDCPInstanceSpecifier();
    s_instance >> m;
    if (s_instance.fail()) invalidDCPInstanceSpecifier();
    if (s_instance.get() != '_') invalidDCPInstanceSpecifier();
    s_instance >> bits_per_variable;
    if (s_instance.fail()) invalidDCPInstanceSpecifier();
    if (s_instance.get() != '_') invalidDCPInstanceSpecifier();
    s_instance >> discretizer;
    if (s_instance.fail()) invalidDCPInstanceSpecifier();

    return {
        dcp_continuous_functions.at(function_name),
        m,
        bits_per_variable,
        discretizer
    };
}

DiscretizedContinousProblem discretizedContinuousProblem;
void initializeDiscretizedContinuousProblem()
{
    assert(instance.has_value());
    discretizedContinuousProblem = loadDiscretizedContinuousProblem(instance.value());
    assert(discretizedContinuousProblem.bits_per_variable * discretizedContinuousProblem.m == number_of_parameters);
    assert(number_of_objectives == 2);
    
    optimization = (char*)Malloc(number_of_objectives * sizeof(char));
    for (int o = 0; o < number_of_objectives; ++o)
    {
        optimization[o] = MAXIMIZATION;
    }
}

double* decodeBitsToDoubleDCP(DiscretizedContinousProblem &dcProblem, char *bits)
{
    double* result = (double*) Malloc(dcProblem.m * sizeof(double));
    for (int i = 0; i < dcProblem.m; ++i)
    {
        switch (dcProblem.discretizer)
        {
            case 0:
                result[i] = decodeBinarySpaced01(&bits[i * dcProblem.bits_per_variable], dcProblem.bits_per_variable);
                break;
            case 1:
                result[i] = decodeGraySpaced01(&bits[i * dcProblem.bits_per_variable], dcProblem.bits_per_variable);
                break;
            default:
                std::cerr << "Unknown discretizer " << dcProblem.discretizer << ".\n";
                std::exit(1);
        }
    }
    return result;
}

void DCProblemEvaluation( char *solution, double *obj_values, double *con_value, int /* objective_index_of_extreme_cluster */ )
{
    double* continuous_solution = decodeBitsToDoubleDCP(discretizedContinuousProblem, solution);
    discretizedContinuousProblem.function(discretizedContinuousProblem.m, continuous_solution, obj_values);
    *con_value = 0;
    free(continuous_solution);
}

/* END OF PROBLEMS */

/* Generic front override - code */

void invalidOverrideFrontFile(std::ifstream &file, std::string expected)
{
    std::cerr << "Override front provided (" << overrideFrontPath.value() << ") is not valid.\n";
    std::cerr << "Was unable to read at position " << file.tellg() << ".\n";
    std::cerr << expected;
    std::cerr << std::endl;
    exit(1);
}

void loadOverrideFront()
{
    if (! overrideFrontPath.has_value()) return;
    std::ifstream file(overrideFrontPath.value());
    
    int number_of_objectives = 0;
    int number_of_points = 0;
    std::optional<std::string> instance;
    std::string maybe_instance;

    file >> number_of_objectives;
    if(file.fail()) invalidOverrideFrontFile(file, "expected an integer (number_of_objectives)");
    file >> number_of_points;
    if(file.fail()) invalidOverrideFrontFile(file, "expected an integer (number_of_points)");
    // read line end.
    if(! std::getline(file, maybe_instance)) invalidOverrideFrontFile(file,  "expected an newline");
    // next line.
    if(! std::getline(file, maybe_instance)) invalidOverrideFrontFile(file, "expected an string (instance specifier)");

    // https://stackoverflow.com/questions/216823/how-to-trim-a-stdstring
    const char* ws = " \t\n\r\f\v";
    maybe_instance.erase(maybe_instance.find_last_not_of(ws) + 1);    
    if (maybe_instance != "") instance = maybe_instance;

    std::vector<double> objectives;
    objectives.reserve(number_of_objectives * number_of_points);

    double t = 0.0;
    while (file >> t)
    {
        objectives.push_back(t);
    }
    // Ensure matrix is shaped correctly.
    int num_objective_values = static_cast<int>(objectives.size());
    int num_expected_objective_values = number_of_objectives * number_of_points;
    int num_expected_objective_values_with_reference = number_of_objectives * (number_of_points + 1);
    bool has_reference_point = num_objective_values == num_expected_objective_values_with_reference;
    if (!has_reference_point &&
        num_objective_values != num_expected_objective_values)
        invalidOverrideFrontFile(file, "expected objectives to be shaped correctly.\nExpected " +
                                           std::to_string(num_expected_objective_values) + " or " +
                                           std::to_string(num_expected_objective_values + number_of_objectives) + " got " +
                                           std::to_string(num_objective_values) + ".");

    use_vtr = TRUE;

    overrideFront = OverrideFront {
        number_of_objectives,
        number_of_points,
        instance,
        has_reference_point,
        objectives
    };
}


double **getConvertedOverrideFront( int *default_front_size )
{
    static double **result = NULL;

    *default_front_size = 0;

    if (! overrideFront.has_value()) return NULL;
    
    OverrideFront &front = overrideFront.value();
    *default_front_size = front.number_of_points;

    if (result != NULL) return result;

    assert(number_of_objectives == front.number_of_objectives);
    // If front specifies a non-empty instance value, check it to ensure we are always using a front corresponding to the right problem.
    if (front.instance.has_value() && instance.has_value() && instance != front.instance)
    {
        std::cerr << "Error: instance set (" << instance.value() << ") and" << std::endl << 
            "the instance for which this front is determined (" << front.instance.value() << ")" << std::endl << " are not the same." << std::endl;
        assert(instance == front.instance);
    }
    
    assert(front.number_of_points * front.number_of_objectives <= static_cast<int>(front.objectives.size()));
    
    // Copy such that freeing does not double (or n-tuple) free the memory.
    result = (double**) Malloc(*default_front_size * sizeof(double*));
    for (int p = 0; p < *default_front_size; ++p)
    {
        result[p] = (double*) Malloc(number_of_objectives * sizeof(double));
        for (int o = 0; o < number_of_objectives; ++o)
        {
            result[p][o] = front.objectives[p * number_of_objectives + o];
        }
    }

    return result;
}

/**
 * Returns whether the D_{Pf->S} metric can be computed.
 */
short haveDPFSMetric( void )
{
    int default_front_size;

    getDefaultFront( &default_front_size );
    if( default_front_size > 0 )
        return( 1 );

    return( 0 );
}


/**
 * Returns the default front(NULL if there is none).
 * The number of solutions in the default
 * front is returned in the pointer variable.
 */
double **getDefaultFront( int *default_front_size )
{
    // Override the problems default front if one is provided.
    if (overrideFront.has_value())
    {
        return getConvertedOverrideFront( default_front_size );
    }

    switch( problem_index )
    {
        case ZEROMAX_ONEMAX: return( getDefaultFrontOnemaxZeromax( default_front_size ) );
        case TRAP5: return( getDefaultFrontTrap5InverseTrap5( default_front_size ) );
        case LOTZ: return( getDefaultFrontLeadingOneTrailingZero( default_front_size ) );
        case MAXCUT: return( getDefaultFrontMAXCUT( default_front_size ) );
        case MAXSAT: return ( getDefaultFrontMAXSAT( default_front_size ) );
        case BESTOFTRAPS:  return( getDefaultFrontBestOfTraps( default_front_size ) );
    }

    *default_front_size = 0;
    return( NULL );
}

double computeDPFSMetric( double **default_front, int default_front_size, double **approximation_front, int approximation_front_size )
{
    int    i, j;
    double result, distance, smallest_distance;

    if( approximation_front_size == 0 )
        return( 1e+308 );

    result = 0.0;
    for( i = 0; i < default_front_size; i++ )
    {
        smallest_distance = 1e+308;
        for( j = 0; j < approximation_front_size; j++ )
        {
            distance = distanceEuclidean( default_front[i], approximation_front[j], number_of_objectives );
            if( distance < smallest_distance )
                smallest_distance = distance;
      
        }
        result += smallest_distance;
    }
    result /= (double) default_front_size;

    return( result );
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=- Tracking Progress =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Writes (appends) statistics about the current generation to a
 * file named "statistics.dat".
 */
void writeGenerationalStatistics( void )
{
    int     i;
    std::ofstream file;

    // Skip computing statistics (save the harddrive & time)
    if(! write_statistics_dat_file) return;

    if(( number_of_generations == 0 && population_id == 0) ||
        (number_of_generations == 0 && population_id == -1))
    {
        file.open(outpath / "statistics.dat", std::ios::out);
        file << "# Generation,Population Size,Evaluations,Time (ms)";

        if (canComputeHypervolume())
            file << ",hv";
        if (write_clusters_with_statistics)
            file << ",[ Cluster_Index ]";
        file << '\n';
    }
    else
        file.open(outpath / "statistics.dat", std::ios::app);

    std::chrono::duration time_since_start = std::chrono::system_clock::now() - time_at_start;
    int millis_since_start = std::chrono::duration_cast<std::chrono::milliseconds>(time_since_start).count();

    file << std::setprecision(10)
        << number_of_generations  << ","
        << population_size << ","
        << std::setprecision(11) << number_of_evaluations << ","
        << millis_since_start;
    
    if (canComputeHypervolume())
            file << ',' << computeCurrentHypervolume();

    if (write_clusters_with_statistics)
    {
        file << ',' << '[';
        file << std::setprecision(4);
        for( i = 0; i < number_of_mixing_components; i++ )
        {
            file << i;
            // Skip writing cluster sizes at generation 0: generation 0 is only initialization, does not perform clustering.
            if (population_cluster_sizes != NULL && number_of_generations != 0)
                file << "(" << population_cluster_sizes[i] << ")";
            if( i < number_of_mixing_components-1 )
            {
                file << ";";
            }
        }
        file << ']';
    }
    file << '\n';

    file.close();
}

bool first_time_writing_hv = true;

void writeCurrentHypervolume(bool writeAtLastImprovement)
{
    // Skip if Hypervolume cannot be computed, or writing is disabled.
    if (!write_hypervolume || !canComputeHypervolume()) return;

    static long last_archive_improvement_written = -1;
    
    // No change in hypervolume to be expected, archive hasn't changed.
    if (last_archive_improvement_written == number_of_evaluations_at_last_archive_improvement) return;
    last_archive_improvement_written = number_of_evaluations_at_last_archive_improvement;

    std::ofstream file;
    if (first_time_writing_hv)
    {
        first_time_writing_hv = false;
        file.open(outpath / "hypervolume.dat", std::ios::out);
        file << "#evaluations,time (ms),hypervolume\n";
    }
    else
    {
        file.open(outpath / "hypervolume.dat", std::ios::app);
    }

    long num_evals = number_of_evaluations;
    std::chrono::duration time_since_start = std::chrono::system_clock::now() - time_at_start;
    if (writeAtLastImprovement)
    {
        num_evals = number_of_evaluations_at_last_archive_improvement;
        time_since_start = time_at_last_archive_improvement - time_at_start;
    }
    int millis_since_start = std::chrono::duration_cast<std::chrono::milliseconds>(time_since_start).count();

    file << std::setprecision(11) << num_evals << ',' << millis_since_start << ',' << computeCurrentHypervolume() << '\n';

    file.close();
}

void writeCurrentPopulation( char kind )
{
    std::ofstream file;
    bool with_line = (((int) lines.size()) == number_of_mixing_components) && (((int) nearest_line.size()) == population_size);

    std::string filename;

    if (! write_population) return;
    
    if (keep_only_latest_population)
        filename = "population.dat.tmp";
    else if (kind == 1) // kind 1 = final
        filename = "population_final.dat";
    else if (kind == 2) // kind 2 = init
        filename = "population_init.dat";
    else // kind 0 = usual
        filename = "population_at_evaluation_" + std::to_string(number_of_evaluations) + ".dat";

    auto filepath = outpath / filename;
    file.open(filepath, std::ios::out);

    // Header: objectives
    // file << "cluster" << ",";
    file << "population_id" << "," << "current" << ",";
    if (with_line)
    {
        file << "nearest_line" << ",";
        file << "assigned_line" << ",";
    }
    for(int j = 0; j < number_of_objectives; j++ )
    {
        file << "objective" << j << ",";
    }
    file << "constraint" << ",";
    file << "solution" << ",";
    file << "is_extreme";

    // Note: this writes its own comma.
    writeCSVHeaderMetadata(file, true);

    file << '\n';

    for (int w_pop_id = 0; w_pop_id < number_of_populations; ++w_pop_id)
    {
        if (w_pop_id == population_id)
        {
            // Write current population, which may be in progress.

            // Note: uses initialized_population_size as this function may be called during initialization,
            //       at which point the solutions after this point have not been initialized yet.
            for (int i = 0; i < initialized_population_size; ++i)
            {
                // Write population id
                file << w_pop_id << ",";
                // is current population.
                file << 1 << ",";
                bool use_offspring = write_offspring_halfway_generation;
                // Only write offspring up to the point initialized.
                use_offspring = use_offspring && i < initialized_offspring_size;

                if (with_line)
                {
                    file << nearest_line[i] << ",";
                    file << assigned_line[i] << ",";
                }
                for(int o = 0; o < number_of_objectives; o++ )
                {
                    if (use_offspring)
                        file << objective_values_offspring[i][o] << ",";
                    else
                        file << objective_values[i][o] << ",";
                }
                file << constraint_values[i] << ",";
                for(int v = 0; v < number_of_parameters; ++v)
                {
                    if (use_offspring)
                        file << (int) offspring[i][v];
                    else
                        file << (int) population[i][v];
                }
                file << ",";
                if (number_of_generations > 0)
                    file << is_extreme_kernel[i];
                else
                    file << "-2";
                if (use_offspring)
                    writeCSVMetadata(file, solution_metadata_offspring[i], true);
                else
                    writeCSVMetadata(file, solution_metadata[i], true);
                file << "\n";
            }
        }
        else
        {
            // Write subpopulation
            int inactive_population_size = array_of_population_sizes[w_pop_id];

            for (int i = 0; i < inactive_population_size; ++i)
            {                
                // Write population id
                file << w_pop_id << ",";
                // is not the current population.
                file << 0 << ",";
                
                if (with_line)
                {
                    // Lines are undefined -- they are not stored across generations.
                    file << -1 << ",";
                    file << -1 << ",";
                }
                for(int o = 0; o < number_of_objectives; o++ )
                {
                    file << array_of_objective_values[w_pop_id][i][o] << ",";
                }
                file << array_of_constraint_values[w_pop_id][i] << ",";
                for(int v = 0; v < number_of_parameters; ++v)
                {
                    file << (int) array_of_populations[w_pop_id][i][v];
                }
                file << ",-2";

                writeCSVMetadata(file, array_of_solution_metadatas[w_pop_id][i], true);
                file << "\n";
            }
        }
    }

    file.close();
    if (keep_only_latest_population)
        std::filesystem::rename(filepath, outpath / "population.dat");
}

void writeCurrentLines( char final )
{
    bool with_line = (((int) lines.size()) == number_of_mixing_components) && (((int) nearest_line.size()) == population_size);
    // Lines are not calculated, return
    if (!with_line) return;

    std::ofstream file;

    std::string filename;

    if (keep_only_latest_lines)
        filename = "lines.dat.tmp";
    if (final)
        filename = "lines_final.dat";
    else
        filename = "lines_at_evaluation_" + std::to_string(number_of_evaluations) + ".dat";

    auto filepath = outpath / filename;
    file.open(filepath, std::ios::out);

    // Header: objectives
    for(int j = 0; j < number_of_objectives; j++ )
    {
        file << "l1_objective" << j << ",";
        file << "l2_objective" << j << ",";
    }
    file << '\n';
    

    for (size_t i = 0; i < lines.size(); ++i)
    {
        for(int o = 0; o < number_of_objectives; o++ )
        {
            file << lines[i].first[o] << ",";
            file << lines[i].second[o] << ",";
        }
        file << "\n";
    }

    file.close();
    if (keep_only_latest_lines)
        std::filesystem::rename(filepath, outpath / "lines.dat");
}

void writeCurrentElitistArchive( char final )
{
    int   i, j;
    std::ofstream file;

    if (!write_elitist_archive) return;

    std::string filename;
    /* Elitist archive */
    if (keep_only_latest_elitist_archive)
        filename = "elitist_archive.dat.tmp";
    if( final )
        filename = "elitist_archive_generation_final.dat";
    else
        filename = "elitist_archive_at_evaluation_" + std::to_string(number_of_evaluations) + ".dat";

    auto filepath = outpath / filename;
    file.open(filepath, std::ios::out);

    // Header: objectives
    for( j = 0; j < number_of_objectives; j++ )
    {
        file << "objective" << j << ",";
    }
    file << "constraint" << ",";
    file << "solution";

    writeCSVHeaderMetadata(file, true);

    file << "\n";

    for( i = 0; i < elitist_archive_size; i++ )
    {
        for( j = 0; j < number_of_objectives; j++ )
        {
            file << std::setprecision(13) << elitist_archive_objective_values[i][j] << ",";
        }

        file << elitist_archive_constraint_values[i] << ",";

        for( j = 0; j < number_of_parameters; j++ )
        {
            file << (int) elitist_archive[i][j]; // << " " - for spacing
        }

        writeCSVMetadata(file, elitist_archive_solution_metadata[i], true);

        file << "\n";
    }
    file.close();

    // After writing, replace the original file.
    if (keep_only_latest_elitist_archive)
        std::filesystem::rename(filepath, outpath / "elitist_archive.dat");
}

void logElitistArchiveAtSpecificPoints()
{
    if(number_of_evaluations%log_progress_interval == 0)
    {
        writeCurrentElitistArchive( FALSE );
        writeCurrentLines( FALSE );
        writeCurrentPopulation( FALSE );
        writeCurrentHypervolume( write_last_archive_improvement_always );
    }
}
char checkAllPopulationsTerminatedCondition()
{
    // If not all populations are initialized yet, neither are they terminated.
    if (number_of_populations != maximum_number_of_populations && allow_starting_of_more_than_one_population )
        return FALSE;

    // The first population is allowed to be started (even with the flag turned on.)
    if (!allow_starting_of_more_than_one_population && number_of_populations < 1)
        return FALSE;

    int max_population_id = maximum_number_of_populations;
    // If starting new populations is disallowed, the current number of populations is the actual maximum.
    if (! allow_starting_of_more_than_one_population)
        max_population_id = number_of_populations;

    for (int population_id = 0; population_id < max_population_id; ++max_population_id)
    {
        if(array_of_population_statuses[population_id] == TRUE)
        {
            return FALSE;
        }
    }

    return TRUE;
}
/**
 * Returns TRUE if termination should be enforced, FALSE otherwise.
 */
char checkTerminationCondition()
{
    if( maximum_number_of_evaluations >= 0 )
    {
        if( checkNumberOfEvaluationsTerminationCondition() )
            return( TRUE );
    }

    if( use_vtr )
    {
        if( checkVTRTerminationCondition() )
            return( TRUE );
    }

    if (checkAllPopulationsTerminatedCondition())
        return TRUE;

    return( FALSE );
}
char checkPopulationConverged()
{
    for (int i = 0; i < population_size; ++i)
    {
        for (int d = 0; d < number_of_parameters; ++d)
        {
            // Found counterexample to a converged population.
            if (population[i][d] != population[0][d]) return FALSE;
        }
    }
    return TRUE;
}
/**
 * Returns TRUE if the current population should be terminated, FALSE otherwise.
 */
char checkPopulationTerminationCriterion()
{
    if (terminate_population_upon_convergence && checkPopulationConverged())
        return TRUE;

    return FALSE;
}
/**
 * Returns TRUE if the maximum number of evaluations
 * has been reached, FALSE otherwise.
 */
char checkNumberOfEvaluationsTerminationCondition()
{
  if( number_of_evaluations >= maximum_number_of_evaluations )
    return( TRUE );

  return( FALSE );
}
/**
 * Returns 1 if the value-to-reach has been reached
 * for the multi-objective case. This means that
 * the D_Pf->S metric has reached the value-to-reach.
 * If no D_Pf->S can be computed, 0 is returned.
 */
char checkVTRTerminationCondition( void )
{
  int      default_front_size;
  double **default_front, metric_elitist_archive;

  if( haveDPFSMetric() )
  {
    default_front          = getDefaultFront( &default_front_size );
    metric_elitist_archive = computeDPFSMetric( default_front, default_front_size, elitist_archive_objective_values, elitist_archive_size );

    if( metric_elitist_archive <= vtr )
    {
      return( 1 );
    }
  }

  return( 0 );
}

void logNumberOfEvaluationsAtVTR()
{
    int      default_front_size;
    double **default_front, metric_elitist_archive;
    std::ofstream file;

    if(use_vtr == FALSE)
        return;

    if( haveDPFSMetric() )
    {
        default_front          = getDefaultFront( &default_front_size );
        metric_elitist_archive = computeDPFSMetric( default_front, default_front_size, elitist_archive_objective_values, elitist_archive_size );

        std::string filename = "number_of_evaluations_when_all_points_found.dat";
        // std::string filename = "number_of_evaluations_when_all_points_found_" + std::to_string(number_of_parameters) + ".dat";
        file.open(outpath / filename, std::ios::app);
        if( metric_elitist_archive <= vtr )
        {
            // file << number_of_evaluations << "\n";
            file << number_of_evaluations_at_last_archive_improvement << "\n";
        }
        else
        {
            file << "NA" << "\n";  
        }
        file.close();
    }
}

/*=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=  Elitist Archive -==-=-=-=-=-=-=-=-=-=-=-=*/
char isDominatedByElitistArchive( double *obj, double con, char *is_new_nondominated_point, int *position_of_existed_member )
{
    int j;

    *is_new_nondominated_point = TRUE;
    *position_of_existed_member = -1;
    for( j = 0; j < elitist_archive_size; j++ )
    {
        if( constraintParetoDominates( elitist_archive_objective_values[j], elitist_archive_constraint_values[j], obj, con ) )
        {
            *is_new_nondominated_point = FALSE;
            return( TRUE );
        }
        else
        {
            if( !constraintParetoDominates( obj, con, elitist_archive_objective_values[j], elitist_archive_constraint_values[j] ) )
            {
              if( sameObjectiveBox( elitist_archive_objective_values[j], obj ) )
              {
                *is_new_nondominated_point = FALSE;
                *position_of_existed_member = j;
                return( FALSE );
              }
            }
        }
    }
    return( FALSE );
}
/**
 * Returns 1 if two solutions share the same objective box, 0 otherwise.
 */
short sameObjectiveBox( double *objective_values_a, double *objective_values_b )
{
    int i;

    if( !objective_discretization_in_effect )
    {
        /* If the solutions are identical, they are still in the (infinitely small) same objective box. */
        for( i = 0; i < number_of_objectives; i++ )
        {
            if( objective_values_a[i] != objective_values_b[i] )
                return( 0 );
        }

        return( 1 );
    }


    for( i = 0; i < number_of_objectives; i++ )
    {
        if( ((int) (objective_values_a[i] / objective_discretization[i])) != ((int) (objective_values_b[i] / objective_discretization[i])) )
            return( 0 );
    }

    return( 1 );
}

int hammingDistanceInParameterSpace(char *solution_1, char *solution_2)
{
	int i, distance;
	distance=0;
	for (i=0; i < number_of_parameters; i++)
	{
		if( solution_1[i] != solution_2[i])
			distance++;
	}

	return distance;
}

int hammingDistanceToNearestNeighborInParameterSpace(char *solution, int replacement_position)
{
	int i, distance_to_nearest_neighbor, distance;
	distance_to_nearest_neighbor = -1;
	for (i = 0; i < elitist_archive_size; i++)
	{
		if (i != replacement_position)
		{
			distance = hammingDistanceInParameterSpace(solution, elitist_archive[i]);
			if (distance < distance_to_nearest_neighbor || distance_to_nearest_neighbor < 0)
				distance_to_nearest_neighbor = distance;
		}
	}

	return distance_to_nearest_neighbor;
}
/**
 * Updates the elitist archive by offering a new solution
 * to possibly be added to the archive. If there are no
 * solutions in the archive yet, the solution is added.
 * Solution A is always dominated by solution B that is
 * in the same domination-box if B dominates A or A and
 * B do not dominate each other. If the solution is not
 * dominated, it is added to the archive and all solutions
 * dominated by the new solution, are purged from the archive.
 */
void updateElitistArchive( char *solution, double *solution_objective_values, double solution_constraint_value, void* solution_metadata)
{
    short is_dominated_itself;
    int   i, *indices_dominated, number_of_solutions_dominated;

    if( elitist_archive_size == 0 )
        addToElitistArchive( solution, solution_objective_values, solution_constraint_value, solution_metadata);
    else
    {
        indices_dominated             = (int *) Malloc( elitist_archive_size*sizeof( int ) );
        number_of_solutions_dominated = 0;
        is_dominated_itself           = 0;
        for( i = 0; i < elitist_archive_size; i++ )
        {
            if( constraintParetoDominates( elitist_archive_objective_values[i], elitist_archive_constraint_values[i], solution_objective_values, solution_constraint_value ) )
                is_dominated_itself = 1;
            else
            {
                if( !constraintParetoDominates( solution_objective_values, solution_constraint_value, elitist_archive_objective_values[i], elitist_archive_constraint_values[i] ) )
                {
                    if( sameObjectiveBox( elitist_archive_objective_values[i], solution_objective_values ) )
                        is_dominated_itself = 1;
                }
            }

            if( is_dominated_itself )
                break;
        }

        if( !is_dominated_itself )
        {
            for( i = 0; i < elitist_archive_size; i++ )
            {
                if( constraintParetoDominates( solution_objective_values, solution_constraint_value, elitist_archive_objective_values[i], elitist_archive_constraint_values[i] ) )
                {
                    indices_dominated[number_of_solutions_dominated] = i;
                    number_of_solutions_dominated++;
                }
            }

            if( number_of_solutions_dominated > 0 )
                removeFromElitistArchive( indices_dominated, number_of_solutions_dominated );

            addToElitistArchive( solution, solution_objective_values, solution_constraint_value, solution_metadata);
        }

        free( indices_dominated );
    }
}
void updateElitistArchiveWithReplacementOfExistedMember( char *solution, double *solution_objective_values, double solution_constraint_value, void* solution_metadata, char *is_new_nondominated_point, char *is_dominated_by_archive)
{
    short is_existed, index_of_existed_member;
    int   i, *indices_dominated, number_of_solutions_dominated;
    int distance_old, distance_new;

    *is_new_nondominated_point  = TRUE;
    *is_dominated_by_archive    = FALSE;

    if( elitist_archive_size == 0 )
        addToElitistArchive( solution, solution_objective_values, solution_constraint_value, solution_metadata);
    else
    {
        indices_dominated             = (int *) Malloc( elitist_archive_size*sizeof( int ) );
        number_of_solutions_dominated = 0;
        is_existed					  = 0;
        for( i = 0; i < elitist_archive_size; i++ )
        {
            if( constraintParetoDominates( elitist_archive_objective_values[i], elitist_archive_constraint_values[i], solution_objective_values, solution_constraint_value ) )
            {
                *is_dominated_by_archive    = TRUE;
                *is_new_nondominated_point  = FALSE;
            }
            else
            {
                if( !constraintParetoDominates( solution_objective_values, solution_constraint_value, elitist_archive_objective_values[i], elitist_archive_constraint_values[i] ) )
                {
                    if( sameObjectiveBox( elitist_archive_objective_values[i], solution_objective_values ) )
                    {
                        is_existed                  = 1;
                        index_of_existed_member     = i;
                        *is_new_nondominated_point  = FALSE;
                    }
                }
            }

            if( (*is_new_nondominated_point) == FALSE )
                break;
        }

        if( (*is_new_nondominated_point) == TRUE )
        {
            for( i = 0; i < elitist_archive_size; i++ )
            {
                if( constraintParetoDominates( solution_objective_values, solution_constraint_value, elitist_archive_objective_values[i], elitist_archive_constraint_values[i] ) )
                {
                    indices_dominated[number_of_solutions_dominated] = i;
                    number_of_solutions_dominated++;
                }
            }

            if( number_of_solutions_dominated > 0 )
                removeFromElitistArchive( indices_dominated, number_of_solutions_dominated );

            addToElitistArchive( solution, solution_objective_values, solution_constraint_value, solution_metadata);
            elitist_archive_front_changed = TRUE;
        }

        if( is_existed )
        {
            distance_old = hammingDistanceToNearestNeighborInParameterSpace(elitist_archive[index_of_existed_member], index_of_existed_member);
            distance_new = hammingDistanceToNearestNeighborInParameterSpace(solution, index_of_existed_member);

            if (distance_new > distance_old)
            {
                for(i = 0; i < number_of_parameters; i++)
                    elitist_archive[index_of_existed_member][i] = solution[i];
                for(i=0; i < number_of_objectives; i++)
                    elitist_archive_objective_values[index_of_existed_member][i] = solution_objective_values[i];
                elitist_archive_constraint_values[index_of_existed_member] = solution_constraint_value;
                copySolutionMetadata(
                /* from */ solution_metadata, 
                /*   to */ elitist_archive_solution_metadata[index_of_existed_member]);
            }
        }
    
        free( indices_dominated );
    }
}

/**
 * Removes a set of solutions (identified by their archive-indices)
 * from the elitist archive.
 */
void removeFromElitistArchive( int *indices, int number_of_indices )
{
    int      i, j, elitist_archive_size_new;
    char **elitist_archive_new;
    double **elitist_archive_objective_values_new;
	double *elitist_archive_constraint_values_new;
    void   **elitist_archive_solution_metadata_new;

    elitist_archive_new                   = (char**) Malloc( elitist_archive_capacity*sizeof( char * ) );
    elitist_archive_objective_values_new  = (double **) Malloc( elitist_archive_capacity*sizeof( double * ) );
    elitist_archive_constraint_values_new = (double *) Malloc( elitist_archive_capacity*sizeof( double ) );
    elitist_archive_solution_metadata_new = (void**) Malloc( elitist_archive_capacity*sizeof( void * ) );

    for( i = 0; i < elitist_archive_capacity; i++ )
    {
        elitist_archive_new[i]                  = (char *) Malloc( number_of_parameters*sizeof( char ) );
        elitist_archive_objective_values_new[i] = (double *) Malloc( number_of_objectives*sizeof( double ) );
        initSolutionMetadata(elitist_archive_solution_metadata_new[i]);
    }

    elitist_archive_size_new = 0;
    for( i = 0; i < elitist_archive_size; i++ )
    {
        if( !isInListOfIndices( i, indices, number_of_indices ) )
        {
            for( j = 0; j < number_of_parameters; j++ )
                elitist_archive_new[elitist_archive_size_new][j] = elitist_archive[i][j];
            for( j = 0; j < number_of_objectives; j++ )
                elitist_archive_objective_values_new[elitist_archive_size_new][j] = elitist_archive_objective_values[i][j];
            elitist_archive_constraint_values_new[elitist_archive_size_new] = elitist_archive_constraint_values[i];
            copySolutionMetadata(
                /* from */ elitist_archive_solution_metadata[i], 
                /*   to */ elitist_archive_solution_metadata_new[elitist_archive_size_new]);

            elitist_archive_size_new++;
        }
    }

    for( i = 0; i < elitist_archive_capacity; i++ )
    {
        free( elitist_archive[i] );
        free( elitist_archive_objective_values[i] );
        cleanupSolutionMetadata(elitist_archive_solution_metadata[i]);
    }
    free( elitist_archive );
    free( elitist_archive_objective_values );
    free( elitist_archive_constraint_values );
    free( elitist_archive_solution_metadata );

    elitist_archive_size              = elitist_archive_size_new;
    elitist_archive                   = elitist_archive_new;
    elitist_archive_objective_values  = elitist_archive_objective_values_new;
    elitist_archive_constraint_values = elitist_archive_constraint_values_new;
    elitist_archive_solution_metadata = elitist_archive_solution_metadata_new;
}

/**
 * Returns 1 if index is in the indices array, 0 otherwise.
 */
short isInListOfIndices( int index, int *indices, int number_of_indices )
{
    int i;

    for( i = 0; i < number_of_indices; i++ )
        if( indices[i] == index )
        return( 1 );

    return( 0 );
}

/**
 * Adds a solution to the elitist archive.
 */
void addToElitistArchive( char *solution, double *solution_objective_values, double solution_constraint_value, void* solution_metadata )
{
    int      i, j, elitist_archive_capacity_new;
    char **elitist_archive_new;
    double **elitist_archive_objective_values_new;
	double *elitist_archive_constraint_values_new;
	void   **elitist_archive_solution_metadata_new;
    

    if( elitist_archive_capacity == elitist_archive_size )
    {
        elitist_archive_capacity_new          = elitist_archive_capacity*2+1;
        elitist_archive_new                   = (char **) Malloc( elitist_archive_capacity_new*sizeof( char * ) );
        elitist_archive_objective_values_new  = (double **) Malloc( elitist_archive_capacity_new*sizeof( double * ) );
        elitist_archive_constraint_values_new = (double *) Malloc( elitist_archive_capacity_new*sizeof( double ) );
        elitist_archive_solution_metadata_new = (void**) Malloc( elitist_archive_capacity_new*sizeof( void * ) );

        for( i = 0; i < elitist_archive_capacity_new; i++ )
        {
            elitist_archive_new[i]                    = (char *) Malloc( number_of_parameters*sizeof( char ) );
            elitist_archive_objective_values_new[i]   = (double *) Malloc( number_of_objectives*sizeof( double ) );
            initSolutionMetadata(elitist_archive_solution_metadata_new[i]);
        }

        for( i = 0; i < elitist_archive_size; i++ )
        {
            for( j = 0; j < number_of_parameters; j++ )
                elitist_archive_new[i][j] = elitist_archive[i][j];
            for( j = 0; j < number_of_objectives; j++ )
                elitist_archive_objective_values_new[i][j] = elitist_archive_objective_values[i][j];
            elitist_archive_constraint_values_new[i] = elitist_archive_constraint_values[i];
            copySolutionMetadata(
                /* from */ elitist_archive_solution_metadata[i], 
                /*   to */ elitist_archive_solution_metadata_new[i]);
        }

        for( i = 0; i < elitist_archive_capacity; i++ )
        {
            free( elitist_archive[i] );
            free( elitist_archive_objective_values[i] );
            cleanupSolutionMetadata(elitist_archive_solution_metadata[i]);
        }
        free( elitist_archive );
        free( elitist_archive_objective_values );
        free( elitist_archive_constraint_values );
        free( elitist_archive_solution_metadata );

        elitist_archive_capacity          = elitist_archive_capacity_new;
        elitist_archive                   = elitist_archive_new;
        elitist_archive_objective_values  = elitist_archive_objective_values_new;
        elitist_archive_constraint_values = elitist_archive_constraint_values_new;
        elitist_archive_solution_metadata = elitist_archive_solution_metadata_new;
    }

    for( j = 0; j < number_of_parameters; j++ )
        elitist_archive[elitist_archive_size][j] = solution[j];
    for( j = 0; j < number_of_objectives; j++ )
        elitist_archive_objective_values[elitist_archive_size][j] = solution_objective_values[j];
    elitist_archive_constraint_values[elitist_archive_size] = solution_constraint_value; // Notice here //
    copySolutionMetadata(solution_metadata, elitist_archive_solution_metadata[elitist_archive_size]);

    number_of_evaluations_at_last_archive_improvement = number_of_evaluations;
    time_at_last_archive_improvement = std::chrono::system_clock::now();
    elitist_archive_size++;
}
/**
 * Adapts the objective box discretization. If the numbre
 * of solutions in the elitist archive is too high or too low
 * compared to the population size, the objective box
 * discretization is adjusted accordingly. In doing so, the
 * entire elitist archive is first emptied and then refilled.
 */
void adaptObjectiveDiscretization( void )
{
    int    i, j, k, na, nb, nc, elitist_archive_size_target_lower_bound, elitist_archive_size_target_upper_bound;
    double low, high, *elitist_archive_objective_ranges;

    elitist_archive_size_target_lower_bound = (int) (0.75*elitist_archive_size_target);
    elitist_archive_size_target_upper_bound = (int) (1.25*elitist_archive_size_target);

    if( objective_discretization_in_effect && (elitist_archive_size < elitist_archive_size_target_lower_bound) )
        objective_discretization_in_effect = 0;

    if( elitist_archive_size > elitist_archive_size_target_upper_bound )
    {
        objective_discretization_in_effect = 1;

        elitist_archive_objective_ranges = (double *) Malloc( number_of_objectives*sizeof( double ) );
        for( j = 0; j < number_of_objectives; j++ )
        {
            low  = elitist_archive_objective_values[0][j];
            high = elitist_archive_objective_values[0][j];

            for( i = 0; i < elitist_archive_size; i++ )
            {
                if( elitist_archive_objective_values[i][j] < low )
                    low = elitist_archive_objective_values[i][j];
                if( elitist_archive_objective_values[i][j] > high )
                    high = elitist_archive_objective_values[i][j];
            }

            elitist_archive_objective_ranges[j] = high - low;
        }

        na = 1;
        nb = (int) pow(2.0,25.0);
        
        for( k = 0; k < 25; k++ )
        {
            nc = (na + nb) / 2;
            for( i = 0; i < number_of_objectives; i++ )
                objective_discretization[i] = elitist_archive_objective_ranges[i]/((double) nc);

            /* Restore the original elitist archive after the first cycle in this loop */
            if( k > 0 )
            {
                elitist_archive_size = 0;
                for( i = 0; i < elitist_archive_copy_size; i++ )
                    addToElitistArchive( elitist_archive_copy[i], elitist_archive_copy_objective_values[i], elitist_archive_copy_constraint_values[i], elitist_archive_copy_solution_metadata[i] );
            }

            /* Copy the entire elitist archive */
            if( elitist_archive_copy != NULL )
            {
                for( i = 0; i < elitist_archive_copy_size; i++ )
                {
                    free( elitist_archive_copy[i] );
                    free( elitist_archive_copy_objective_values[i] );
                    cleanupSolutionMetadata(elitist_archive_copy_solution_metadata[i]);
                }
                free( elitist_archive_copy );
                free( elitist_archive_copy_objective_values );
                free( elitist_archive_copy_constraint_values );
                free( elitist_archive_copy_solution_metadata );
            }

            elitist_archive_copy_size              = elitist_archive_size;
            elitist_archive_copy                   = (char **) Malloc( elitist_archive_copy_size*sizeof( char * ) );
            elitist_archive_copy_objective_values  = (double **) Malloc( elitist_archive_copy_size*sizeof( double * ) );
            elitist_archive_copy_constraint_values = (double *) Malloc( elitist_archive_copy_size*sizeof( double ) );
            elitist_archive_copy_solution_metadata = (void **) Malloc( elitist_archive_copy_size*sizeof( void * ) );
      
            for( i = 0; i < elitist_archive_copy_size; i++ )
            {
                elitist_archive_copy[i]                  = (char *) Malloc( number_of_parameters*sizeof( char ) );
                elitist_archive_copy_objective_values[i] = (double *) Malloc( number_of_objectives*sizeof( double ) );
                copySolutionMetadata(
                /* from */ elitist_archive_solution_metadata[i], 
                /*   to */ elitist_archive_copy_solution_metadata[i]);
            }
            for( i = 0; i < elitist_archive_copy_size; i++ )
            {
                for( j = 0; j < number_of_parameters; j++ )
                    elitist_archive_copy[i][j] = elitist_archive[i][j];
                for( j = 0; j < number_of_objectives; j++ )
                    elitist_archive_copy_objective_values[i][j] = elitist_archive_objective_values[i][j];
                elitist_archive_copy_constraint_values[i] = elitist_archive_constraint_values[i];
            }

            /* Clear the elitist archive */
            elitist_archive_size = 0;

            /* Rebuild the elitist archive */
            for( i = 0; i < elitist_archive_copy_size; i++ )
                updateElitistArchive( elitist_archive_copy[i], elitist_archive_copy_objective_values[i], elitist_archive_copy_constraint_values[i], elitist_archive_copy_solution_metadata);

            if( elitist_archive_size <= elitist_archive_size_target_lower_bound )
                na = nc;
            else
                nb = nc;
        }

        free( elitist_archive_objective_ranges );
    }
}

/*-=-=-=-=-=-=-=-=-=-=-=-=- Solution Comparison -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
char betterFitness( double *objective_value_x, double constraint_value_x, double *objective_value_y, double constraint_value_y, int objective_index )
{
    short result;

    result = FALSE;

    if( constraint_value_x > 0 ) /* x is infeasible */
    {
        if( constraint_value_y > 0 ) /* Both are infeasible */
        {
            if(constraint_value_x < constraint_value_y)
                result = TRUE;
        }
    }
    else
    {
        if(constraint_value_y > 0)
            result = TRUE;
        else
        {
            if(optimization[objective_index] == MINIMIZATION)
            {
                if(objective_value_x[objective_index] < objective_value_y[objective_index])
                    result = TRUE;
            }
            else if(optimization[objective_index] == MAXIMIZATION) 
            {
                if(objective_value_x[objective_index] > objective_value_y[objective_index])
                    result = TRUE;
            }
        }
    }

    return ( result );
}

char equalFitness(double *objective_value_x, double constraint_value_x, double *objective_value_y, double constraint_value_y, int objective_index )
{
    short result;

    result = FALSE;

    if( constraint_value_x > 0 ) /* x is infeasible */
    {
        if( constraint_value_y > 0 ) /* Both are infeasible */
        {
            if(constraint_value_x == constraint_value_y)
                result = TRUE;
        }
    }
    else
    {
        if(constraint_value_y == 0)
        {
            if(objective_value_x[objective_index] == objective_value_y[objective_index])
                result = TRUE;
        }
    }

    return ( result );
}
/**
 * Returns 1 if x constraint-Pareto-dominates y, 0 otherwise.
 * x is not better than y unless:
 * - x and y are both infeasible and x has a smaller sum of constraint violations, or
 * - x is feasible and y is not, or
 * - x and y are both feasible and x Pareto dominates y
 */
short constraintParetoDominates( double *objective_values_x, double constraint_value_x, double *objective_values_y, double constraint_value_y )
{
    short result;

    result = FALSE;

    if( constraint_value_x > 0 ) /* x is infeasible */
    {
        if( constraint_value_y > 0 ) /* Both are infeasible */
        {
            if( constraint_value_x < constraint_value_y )       
                 result = TRUE;
        }
    }
    else /* x is feasible */
    {
        if( constraint_value_y > 0) /* x is feasible and y is not */
            result = TRUE;
        else /* Both are feasible */
            result = paretoDominates( objective_values_x, objective_values_y );
    }

    return( result );
}

short constraintWeaklyParetoDominates( double *objective_values_x, double constraint_value_x, double *objective_values_y, double constraint_value_y  )
{
    short result;

    result = FALSE;

    if( constraint_value_x > 0 ) /* x is infeasible */
    {
        if( constraint_value_y > 0 ) /* Both are infeasible */
        {
            if(constraint_value_x  <= constraint_value_y )      
                result = TRUE;
        }
    }
    else /* x is feasible */
    {
        if( constraint_value_y > 0 ) /* x is feasible and y is not */
            result = TRUE;
        else /* Both are feasible */
            result = weaklyParetoDominates( objective_values_x, objective_values_y );
    }

    return( result );
}

/**
 * Returns 1 if x Pareto-dominates y, 0 otherwise.
 */
short paretoDominates( double *objective_values_x, double *objective_values_y )
{
    short strict;
    int   i, result;

    result = 1;
    strict = 0;

    for( i = 0; i < number_of_objectives; i++ )
    {
        if( fabs( objective_values_x[i] - objective_values_y[i] ) >= 0.00001 )
        {
            if(optimization[i] == MINIMIZATION)
            {
                if( objective_values_x[i] > objective_values_y[i] )
                {
                    result = 0;
                    break;
                }
                if( objective_values_x[i] < objective_values_y[i] )
                    strict = 1;    
            }
            else if(optimization[i] == MAXIMIZATION)
            {
                if( objective_values_x[i] < objective_values_y[i] )
                {
                    result = 0;
                    break;
                }
                if( objective_values_x[i] > objective_values_y[i] )
                    strict = 1;                    
            }
            
        }
    }

    if( strict == 0 && result == 1 )
        result = 0;

    return( result );
}

short weaklyParetoDominates( double *objective_values_x, double *objective_values_y )
{
    int   i, result;
    result = 1;

    for( i = 0; i < number_of_objectives; i++ )
    {
        if( fabs( objective_values_x[i] - objective_values_y[i] ) >= 0.00001 )
        {
            if(optimization[i] == MINIMIZATION)
            {
                if( objective_values_x[i] > objective_values_y[i] )
                {
                    result = 0;
                    break;
                }
            }
            else if(optimization[i] == MAXIMIZATION)
            {
                if( objective_values_x[i] < objective_values_y[i] )
                {
                    result = 0;
                    break;
                }                
            }

        }
    }
    
    return( result );
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-= Linkage Tree Learning -==-=-=-=-=-=-=-=-=-=-=*/
/**
 * Learn the linkage for a cluster (subpopulation).
 */
void learnLinkageTree( int cluster_index )
{
    char   done;
    int    i, j, k, a, r0, r1, *indices, *order,
         lt_index, factor_size, **mpm_new, *mpm_new_number_of_indices, mpm_new_length,
        *NN_chain, NN_chain_length;
    double p, *cumulative_probabilities, **S_matrix, mul0, mul1;

    // Whether to remove subsets that have no support (i.e. are merged, but there is no linkage that connects the two)
    // i.e. MI between the two subsets is zero.
    // As in:
    //   Thierens, Dirk, and Peter A.N. Bosman. 2013.
    //   ‘Hierarchical Problem Solving with the Linkage Tree Genetic Algorithm’.
    //   In Proceedings of the 15th Annual Conference on Genetic and Evolutionary Computation, 877–84. GECCO ’13.
    //   New York, NY, USA: Association for Computing Machinery.
    //   https://doi.org/10.1145/2463372.2463477.
    bool filter_unsupported = (linkage_mode == 2 || linkage_mode == 3); 
    // Whether to remove subsets that when merged have an estimated MI of 1.
    // This happens if for example a triplet (a, b, c) of variables perfectly describe each other.
    // As in:
    //   Dushatskiy, Arkadiy, Marco Virgolin, Anton Bouter, Dirk Thierens, and Peter A. N. Bosman. 2021.
    //   ‘Parameterless Gene-Pool Optimal Mixing Evolutionary Algorithms’.
    //   ArXiv:2109.05259 [Cs], September. http://arxiv.org/abs/2109.05259.
    bool filter_occluded = (linkage_mode == 2 || linkage_mode == 3);
    // Whether to remove the root of the tree.
    bool filter_root = true;
    // Epsilon for the aforementioned filters.
    // As MI / NMI computation can be imprecise, there should be an allowed margin of error.
    float eps = 1e-6;

    if (linkage_mode == 0 || linkage_mode == 1 || linkage_mode == 2 || linkage_mode == 3)
    {
        /* Compute joint entropy matrix */
        for( i = 0; i < number_of_parameters; i++ )
        {
            for( j = i+1; j < number_of_parameters; j++ )
            {
                indices                  = (int *) Malloc( 2*sizeof( int ) );
                indices[0]               = i;
                indices[1]               = j;
                cumulative_probabilities = estimateParametersForSingleBinaryMarginal( cluster_index, indices, 2, &factor_size );

                MI_matrix[i][j] = 0.0;
                for( k = 0; k < factor_size; k++ )
                {
                    if( k == 0 )
                        p = cumulative_probabilities[k];
                    else
                        p = cumulative_probabilities[k]-cumulative_probabilities[k-1];
                    if( p > 0 )
                        MI_matrix[i][j] += -p*log2(p);
                }

                MI_matrix[j][i] = MI_matrix[i][j];

                free( indices );
                free( cumulative_probabilities );
            }
            indices                  = (int *) Malloc( 1*sizeof( int ) );
            indices[0]               = i;
            cumulative_probabilities = estimateParametersForSingleBinaryMarginal( cluster_index, indices, 1, &factor_size );

            MI_matrix[i][i] = 0.0;
            for( k = 0; k < factor_size; k++ )
            {
                if( k == 0 )
                    p = cumulative_probabilities[k];
                else
                    p = cumulative_probabilities[k]-cumulative_probabilities[k-1];
                if( p > 0 )
                    MI_matrix[i][i] += -p*log2(p);
            }

            free( indices );
            free( cumulative_probabilities );
        }

        /* Then transform into mutual information matrix MI(X,Y)=H(X)+H(Y)-H(X,Y)
        Or the normalized mutual information matrix NMI(X,Y)=(H(X)+H(Y))/H(X,Y) - 1
        Depending on `use_NMI_instead_of_MI` */

        for( i = 0; i < number_of_parameters; i++ )
            for( j = i+1; j < number_of_parameters; j++ )
            {
                if (linkage_mode == 0 || linkage_mode == 2)
                {
                    MI_matrix[i][j] = MI_matrix[i][i] + MI_matrix[j][j] - MI_matrix[i][j];
                }
                else if (linkage_mode == 1 || linkage_mode == 3)
                {
                    double separate = (MI_matrix[i][i] + MI_matrix[j][j]);
                    double joint = MI_matrix[i][j];
                    MI_matrix[i][j] = 0.0;
                    if (joint > 0.0)
                        MI_matrix[i][j] = separate / joint - 1;
                }
                MI_matrix[j][i] = MI_matrix[i][j];
            }
    }
    else if (linkage_mode == -1)
    {
        // Random -- for random tree
        for( i = 0; i < number_of_parameters; i++ )
            for( j = i+1; j < number_of_parameters; j++ )
            {
                MI_matrix[i][j] = randomRealUniform01();
                MI_matrix[j][i] = MI_matrix[i][j];
            }
    }
    else
    {
        std::cerr << "Unknown linkage information mode " << linkage_mode << ".\n";
        std::exit(1);
    }

    /* Initialize MPM to the univariate factorization */
    order                 = createRandomOrdering( number_of_parameters );
    mpm                   = (int **) Malloc( number_of_parameters*sizeof( int * ) );
    mpm_number_of_indices = (int *) Malloc( number_of_parameters*sizeof( int ) );
    mpm_length            = number_of_parameters;
    for( i = 0; i < number_of_parameters; i++ )
    {
        indices                  = (int *) Malloc( 1*sizeof( int ) );
        indices[0]               = order[i];
        mpm[i]                   = indices;
        mpm_number_of_indices[i] = 1;
    }
    free( order );

    /* Initialize LT to the initial MPM */
    if( lt[cluster_index] != NULL )
    {
        for( i = 0; i < lt_length[cluster_index]; i++ )
            free( lt[cluster_index][i] );
        free( lt[cluster_index] );
        free( lt_number_of_indices[cluster_index] );
    }

    // Keep track of which elements in the linkage tree correspond to which element of the mpm.
    int* mpm_to_lt_index = (int*) Malloc(mpm_length * sizeof(int));

    lt[cluster_index]                   = (int **) Malloc( (number_of_parameters+number_of_parameters-1)*sizeof( int * ) );
    lt_number_of_indices[cluster_index] = (int *) Malloc( (number_of_parameters+number_of_parameters-1)*sizeof( int ) );
    lt_length[cluster_index]            = number_of_parameters+number_of_parameters-1;
    lt_index             = 0;
    for( i = 0; i < mpm_length; i++ )
    {
        lt[cluster_index][lt_index]                   = mpm[i];
        lt_number_of_indices[cluster_index][lt_index] = mpm_number_of_indices[i];
        mpm_to_lt_index[i] = lt_index;
        lt_index++;
    }

    // Keep track of an array of which lt elements should be removed as they are occluded
    char* keep_lt = (char *) Malloc( (number_of_parameters+number_of_parameters-1) * sizeof(char) );
    std::fill(keep_lt, keep_lt + number_of_parameters+number_of_parameters-1, TRUE);

    /* Initialize similarity matrix */
    S_matrix = (double **) Malloc( number_of_parameters*sizeof( double * ) );
    for( i = 0; i < number_of_parameters; i++ )
        S_matrix[i] = (double *) Malloc( number_of_parameters*sizeof( double ) );
    for( i = 0; i < mpm_length; i++ )
        for( j = 0; j < mpm_length; j++ )
            S_matrix[i][j] = MI_matrix[mpm[i][0]][mpm[j][0]];
    for( i = 0; i < mpm_length; i++ )
        S_matrix[i][i] = 0;

    NN_chain        = (int *) Malloc( (number_of_parameters+2)*sizeof( int ) );
    NN_chain_length = 0;
    done            = FALSE;
    while( done == FALSE )
    {
        if( NN_chain_length == 0 )
        {
            NN_chain[NN_chain_length] = randomInt( mpm_length );
            NN_chain_length++;
        }

        while( NN_chain_length < 3 )
        {
            NN_chain[NN_chain_length] = determineNearestNeighbour( NN_chain[NN_chain_length-1], S_matrix, mpm_length );
            NN_chain_length++;
        }

        while( NN_chain[NN_chain_length-3] != NN_chain[NN_chain_length-1] )
        {
            NN_chain[NN_chain_length] = determineNearestNeighbour( NN_chain[NN_chain_length-1], S_matrix, mpm_length );
            if( ((S_matrix[NN_chain[NN_chain_length-1]][NN_chain[NN_chain_length]] == S_matrix[NN_chain[NN_chain_length-1]][NN_chain[NN_chain_length-2]])) && (NN_chain[NN_chain_length] != NN_chain[NN_chain_length-2]) )
                NN_chain[NN_chain_length] = NN_chain[NN_chain_length-2];
            NN_chain_length++;
        }
        r0 = NN_chain[NN_chain_length-2];
        r1 = NN_chain[NN_chain_length-1];
        if( r0 > r1 )
        {
            a  = r0;
            r0 = r1;
            r1 = a;
        }
        NN_chain_length -= 3;

        if( r1 < mpm_length ) // This test is required for exceptional cases in which the nearest-neighbor ordering has changed within the chain while merging within that chain
        {
            indices = (int *) Malloc( (mpm_number_of_indices[r0]+mpm_number_of_indices[r1])*sizeof( int ) );
  
            i = 0;
            for( j = 0; j < mpm_number_of_indices[r0]; j++ )
            {
                indices[i] = mpm[r0][j];
                i++;
            }
            for( j = 0; j < mpm_number_of_indices[r1]; j++ )
            {
                indices[i] = mpm[r1][j];
                i++;
            }

            lt[cluster_index][lt_index]                   = indices;
            lt_number_of_indices[cluster_index][lt_index] = mpm_number_of_indices[r0]+mpm_number_of_indices[r1];


            // TODO: It might be possible to apply this one more often if we allow for a single sample that is spread out over the bins
            //       such that the MI is minimized: i.e. can we, with a single (potentially spread out) sample get the mutual information
            //       down to zero.

            // This element should be filtered out afterwards if it is unsupported, and we are filtering unsupported elements.
            // Note: much like another note below, there is some finicky memory management going on that I do not want to touch.
            //       the mpm and the lt share the index subsets, which makes it somewhat annoying to infer.
            bool is_unsupported = (S_matrix[r0][r1] < eps);
            if (filter_unsupported && is_unsupported)
                keep_lt[lt_index] = FALSE;

            // Before we update mpm_to_lt_index, filter the original sets if need be due to occlusion.
            bool is_occluding   = filter_occluded && S_matrix[r0][r1] > 1 - eps;
            if (filter_occluded && is_occluding)
            {
                keep_lt[mpm_to_lt_index[r0]] = FALSE;
                keep_lt[mpm_to_lt_index[r1]] = FALSE;
            }

            // r0 is the new index in the mpm for the merged subset.
            mpm_to_lt_index[r0] = lt_index;
            // r1 and the last mpm index (mpm_length-1) are swapped as r1 is being removed.
            mpm_to_lt_index[r1] = mpm_to_lt_index[mpm_length - 1];
            
            lt_index++;

            // Note: we can filter out unsupported elements from the mpm as well!
            //       as the nearest element was unsupported, all similarities to other subsets are zero.
            //       And therefore the computed similarities for this subset are all zero as well.
            //       However, this implementation is C-style, which makes it very easy to accidentally break
            //       assumptions that have been made.

            mul0 = ((double) mpm_number_of_indices[r0])/((double) mpm_number_of_indices[r0]+mpm_number_of_indices[r1]);
            mul1 = ((double) mpm_number_of_indices[r1])/((double) mpm_number_of_indices[r0]+mpm_number_of_indices[r1]);
            for( i = 0; i < mpm_length; i++ )
            {
                if( (i != r0) && (i != r1) )
                {
                    S_matrix[i][r0] = mul0*S_matrix[i][r0] + mul1*S_matrix[i][r1];
                    S_matrix[r0][i] = S_matrix[i][r0];
                }
            }

            // Copy over the mpm - shinked in size
            mpm_new                   = (int **) Malloc( (mpm_length-1)*sizeof( int * ) );
            mpm_new_number_of_indices = (int *) Malloc( (mpm_length-1)*sizeof( int ) );
            mpm_new_length            = mpm_length-1;
            for( i = 0; i < mpm_new_length; i++ )
            {
                mpm_new[i]                   = mpm[i];
                mpm_new_number_of_indices[i] = mpm_number_of_indices[i];
            }
  
            // Element for r0 is turned into the mpm for the merged subset.
            mpm_new[r0]                   = indices;
            mpm_new_number_of_indices[r0] = mpm_number_of_indices[r0]+mpm_number_of_indices[r1];

            // Element for r1 is swapped with the last mpm index and 'removed'.
            // Note: the index subset that was orginally used for the mpm is also used for the linkage
            // tree above. 
            if( r1 < mpm_length-1 )
            {
                mpm_new[r1]                   = mpm[mpm_length-1];
                mpm_new_number_of_indices[r1] = mpm_number_of_indices[mpm_length-1];
  
                for( i = 0; i < r1; i++ )
                {
                    S_matrix[i][r1] = S_matrix[i][mpm_length-1];
                    S_matrix[r1][i] = S_matrix[i][r1];
                }
  
                for( j = r1+1; j < mpm_new_length; j++ )
                {
                    S_matrix[r1][j] = S_matrix[j][mpm_length-1];
                    S_matrix[j][r1] = S_matrix[r1][j];
                }
            }
  
            for( i = 0; i < NN_chain_length; i++ )
            {
                if( NN_chain[i] == mpm_length-1 )
                {
                    NN_chain[i] = r1;
                    break;
                }
            }
  
            free( mpm );
            free( mpm_number_of_indices );
            mpm                   = mpm_new;
            mpm_number_of_indices = mpm_new_number_of_indices;
            mpm_length            = mpm_new_length;
  
            if( mpm_length == 1 )
                done = TRUE;
        }
    }

    free( NN_chain );

    free( mpm_new );
    free( mpm_number_of_indices );

    // If filtering needs to be performed (i.e. keep_lt does not consist of all trues.)
    if (filter_unsupported || filter_occluded || filter_root)
    {
        int filtered_i = 0;
        int lt_original_length = lt_length[cluster_index];
        for (int i = 0; i < lt_original_length; ++i)
        {
            if (
                (filter_root && i == lt_original_length - 1) ||
                (keep_lt[i] == FALSE)
                )
            {
                // Remove (i.e. free the subset)
                // Pointer will be overwritten by keep, or be in a position
                // larger than lt_length.
                free(lt[cluster_index][i]);
                --lt_length[cluster_index];
            }
            else
            {
                // Keep!
                lt[cluster_index][filtered_i] = lt[cluster_index][i];
                lt_number_of_indices[cluster_index][filtered_i] = lt_number_of_indices[cluster_index][i];
                ++filtered_i;
            }
        }
    }

    free( mpm_to_lt_index );
    free( keep_lt );

    for( i = 0; i < number_of_parameters; i++ )
        free( S_matrix[i] );
    free( S_matrix );
}

/**
 * Estimates the cumulative probability distribution of a
 * single binary marginal for a cluster (subpopulation).
 */
double *estimateParametersForSingleBinaryMarginal( int cluster_index, int *indices, int number_of_indices, int *factor_size )
{
    int     i, j, index, power_of_two;
    char *solution;
    double *result;

    *factor_size = (int) pow( 2, number_of_indices );
    result       = (double *) Malloc( (*factor_size)*sizeof( double ) );

    for( i = 0; i < (*factor_size); i++ )
        result[i] = 0.0;

    for( i = 0; i < population_cluster_sizes[cluster_index]; i++ ) 
    {
        index        = 0;
        power_of_two = 1;
        for( j = number_of_indices-1; j >= 0; j-- )
        {
            solution = population[population_indices_of_cluster_members[cluster_index][i]];
            index += (solution[indices[j]] == TRUE) ? power_of_two : 0;
            power_of_two *= 2;
        }

        result[index] += 1.0;
    }

    for( i = 0; i < (*factor_size); i++ )
        result[i] /= (double) population_cluster_sizes[cluster_index];

    for( i = 1; i < (*factor_size); i++ )
        result[i] += result[i-1];

    result[(*factor_size)-1] = 1.0;

    return( result );
}

/**
 * Determines nearest neighbour according to similarity values.
 */
int determineNearestNeighbour( int index, double **S_matrix, int mpm_length )
{
    int i, result;

    result = 0;
    if( result == index )
        result++;
    for( i = 1; i < mpm_length; i++ )
    {
//    if( (S_matrix[index][i] > S_matrix[index][result]) && (i != index) )
        if( ((S_matrix[index][i] > S_matrix[index][result]) || ((S_matrix[index][i] == S_matrix[index][result]) && (mpm_number_of_indices[i] < mpm_number_of_indices[result]))) && (i != index) )
        result = i;
    }

    return( result );
}

void printLTStructure( int cluster_index )
{
    int i, j;

    for( i = 0; i < lt_length[cluster_index]; i++ )
    {
        printf("[");
        for( j = 0; j < lt_number_of_indices[cluster_index][i]; j++ )
        {
            printf("%d",lt[cluster_index][i][j]);
            if( j < lt_number_of_indices[cluster_index][i]-1 )
                printf(" ");
        }
        printf("]\n");
    }
    printf("\n");
    fflush( stdout );
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- MO-GOMEA -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Performs initializations that are required before starting a run.
 */
void initialize()
{
    number_of_populations++;

    initializeMemory();

    initializePopulationAndFitnessValues();

    computeObjectiveRanges();
}

/**
 * Initializes the memory.
 */
void initializeMemory( void )
{
    int i;

    objective_ranges         = (double *) Malloc( population_size*sizeof( double ) );
    population               = (char **) Malloc( population_size*sizeof( char * ) );
    objective_values         = (double **) Malloc( population_size*sizeof( double * ) );
    constraint_values        = (double *) Malloc( population_size*sizeof( double ) );

    solution_metadata        = (void**) Malloc( population_size * sizeof( void* ) );
    solution_k               = (int*) Malloc( population_size * sizeof( int ) );
    std::fill(solution_k, solution_k + population_size, 0);

    if (use_scalarization)
        tsch_weight_vectors  = computeApproximatelyEquidistantPositiveDirections(population_size, number_of_objectives);
    
    for( i = 0; i < population_size; i++ )
    {
        population[i]        = (char *) Malloc( number_of_parameters*sizeof( char ) );
        objective_values[i]  = (double *) Malloc( number_of_objectives*sizeof( double ) );

        initSolutionMetadata(solution_metadata[i]);
    }

    s_NIS = (int*) Malloc( population_size * sizeof(int) );
    std::fill(s_NIS, s_NIS + population_size, 0);

    t_NIS                    = 0;
    number_of_generations    = 0;
}

int* inversePermutation(int *permutation, int length)
{
    int* result = (int*) Malloc(length * sizeof(int));
    for (int i = 0; i < length; ++i)
        result[permutation[i]] = i;
    return result;
}

/**
 * Initializes the population and the objective values by randomly
 * generation n solutions.
 */
void initializePopulationAndFitnessValues()
{
    int i, j;
    for( i = 0; i < population_size; i++ )
    {
        for( j = 0; j < number_of_parameters; j++ )
            population[i][j] = (randomInt( 2 ) == 1) ? TRUE : FALSE;
        evaluateIndividual( population[i], objective_values[i],  &(constraint_values[i]), solution_metadata[i], NOT_EXTREME_CLUSTER );
        updateElitistArchive( population[i], objective_values[i], constraint_values[i], solution_metadata[i]);    
        initialized_population_size = i + 1;
    }

    double* nadir = NULL;
    double* ideal = NULL;
    double* range = NULL;
    
    switch(initialization_ls_mode)
    {
        case 1: case 2: case 3: case 4: case 5: case 6:
            // Compute and initialize nadir, ideal and range.
            nadir = (double*) Malloc(number_of_objectives * sizeof(double));
            ideal = (double*) Malloc(number_of_objectives * sizeof(double));
            range = (double*) Malloc(number_of_objectives * sizeof(double));
            std::copy(objective_values[0], objective_values[0] + number_of_objectives, nadir);
            std::copy(objective_values[0], objective_values[0] + number_of_objectives, ideal);
            for (int p = 1; p < population_size; ++p)
                for (int o = 0; o < number_of_objectives; ++o)
                {
                    nadir[o] = std::min(objective_values[p][o], nadir[o]);
                    ideal[o] = std::max(objective_values[p][o], ideal[o]);
                }
            for (int o = 0; o < number_of_objectives; ++o)
                range[o] = ideal[o] - nadir[o]; 
    }

    switch(initialization_ls_mode)
    {
        case 1:
            {
            double* weights = (double*) Malloc(number_of_objectives * sizeof(double));
            // Random Direction Hillclimber
            for (int p = 0; p < population_size; ++p)
            {
                // Generate random set of weights for each population member.
                double l = 0;
                for (int o = 0; o < number_of_objectives; ++o)
                {
                    double w = randomRealUniform01();
                    l += w * w;
                    weights[o] = w;
                }
                l = std::sqrt(l);
                for (int o = 0; o < number_of_objectives; ++o)
                    weights[o] /= l;
                // Apply hillclimber
                performHillClimber([&weights, &nadir, &range](
                    char* /* backup */, double* backup_obj, double backup_con,
                    char* /* result */, double*        obj, double        con,
                    bool is_new_undominated_point, bool /* is_dominated_by_archive */)
                {
                    // Accept new undominated points.
                    if (is_new_undominated_point)
                        return true;
                    // Constraint domination.
                    if (backup_con < con)
                        return true;
                    else if (backup_con > con)
                        return false;
                    // Linear Scalarization - maximize 'distance' from nadir \w weights
                    double s_backup = 0.0;
                    double s = 0.0;
                    for (int o = 0; o < number_of_objectives; ++o)
                    {
                        s_backup += (backup_obj[o] - nadir[o]) / range[o] * weights[o];
                        s        += (       obj[o] - nadir[o]) / range[o] * weights[o];
                    }

                    return s >= s_backup;
                },
                    population[p], objective_values[p],  constraint_values[p], solution_metadata[p],
                    population[p], objective_values[p], &constraint_values[p], solution_metadata[p],
                     true, multipass);
            }
            //
            free(nadir); free(ideal); free(range); free(weights);
            }
            break;
        case 2:
            {
            double* weights = (double*) Malloc(number_of_objectives * sizeof(double));
            // Random Direction Hillclimber
            for (int p = 0; p < population_size; ++p)
            {
                // Generate random set of weights for each population member.
                double l = 0;
                for (int o = 0; o < number_of_objectives; ++o)
                {
                    double w = (objective_values[p][o] - nadir[o]) / range[o];
                    l += w * w;
                    weights[o] = w;
                }
                l = std::sqrt(l);
                for (int o = 0; o < number_of_objectives; ++o)
                    weights[o] /= l;
                // Apply hillclimber
                performHillClimber([&weights, &nadir, &range](
                    char* /* backup */, double* backup_obj, double backup_con,
                    char* /* result */, double*        obj, double        con,
                    bool is_new_undominated_point, bool /* is_dominated_by_archive */)
                {
                    // Accept new undominated points.
                    if (is_new_undominated_point)
                        return true;
                    // Constraint domination.
                    if (backup_con < con)
                        return true;
                    else if (backup_con > con)
                        return false;
                    // Linear Scalarization - maximize 'distance' from nadir \w weights
                    double s_backup = 0.0;
                    double s = 0.0;
                    for (int o = 0; o < number_of_objectives; ++o)
                    {
                        s_backup += (backup_obj[o] - nadir[o]) / range[o] * weights[o];
                        s        += (       obj[o] - nadir[o]) / range[o] * weights[o];
                    }

                    return s >= s_backup;
                },
                    population[p], objective_values[p],  constraint_values[p], solution_metadata[p],
                    population[p], objective_values[p], &constraint_values[p], solution_metadata[p],
                     true, multipass);
            }
            //
            free(nadir); free(ideal); free(range); free(weights);
            }
            break;
        case 3: case 4:
            { 
            double* weights = (double*) Malloc(number_of_objectives * sizeof(double));
            bool direction_backup = initialization_ls_mode == 2; 
            for (int p = 0; p < population_size; ++p)
            {
                performHillClimber([&weights, &nadir, &range, &direction_backup](
                    char* /* backup */, double* backup_obj, double backup_con,
                    char* /* result */, double*        obj, double        con,
                    bool is_new_undominated_point, bool /* is_dominated_by_archive */)
                {
                    // Accept new undominated points.
                    if (is_new_undominated_point)
                        return true;
                    // Constraint domination.
                    if (backup_con < con)
                        return true;
                    else if (backup_con > con)
                        return false;
                    
                    // Compute weights for linear scalarization.
                    double l = 0;
                    for (int o = 0; o < number_of_objectives; ++o)
                    {
                        double w;
                        if (direction_backup)
                            w = (backup_obj[o] - nadir[o]) / range[o];
                        else
                            w = (obj[o] - nadir[o]) / range[o];
                        weights[o] = w;
                        l += w * w;
                    }
                    l = std::sqrt(l);
                    for (int o = 0; o < number_of_objectives; ++o)
                        weights[o] /= l;
                    
                    // Linear Scalarization - maximize 'distance' from nadir
                    double s_backup = 0.0;
                    double s = 0.0;
                    for (int o = 0; o < number_of_objectives; ++o)
                    {
                        s_backup += (backup_obj[o] - nadir[o]) / range[o] * weights[o];
                        s        += (       obj[o] - nadir[o]) / range[o] * weights[o];
                    }

                    return s >= s_backup;
                },
                    population[p], objective_values[p],  constraint_values[p], solution_metadata[p],
                    population[p], objective_values[p], &constraint_values[p], solution_metadata[p],
                     true, multipass);
            }
            //
            free(nadir); free(ideal); free(range); free(weights);
            }
            break;
        case 5:
            {
                double ** all_weights = computeApproximatelyEquidistantPositiveDirections(population_size, number_of_objectives);
                int* assignment   = scalarizedAssignWeightVectorsEvenly(number_of_objectives, optimization, population_size, objective_values, population_size, all_weights, range, ideal);
                // int* assignment = createRandomOrdering(population_size);

                for (int p = 0; p < population_size; ++p)
                {
                    double* weights = all_weights[assignment[p]];
                    // Apply hillclimber
                    performHillClimber([&weights, &nadir, &range](
                        char* /* backup */, double* backup_obj, double backup_con,
                        char* /* result */, double*        obj, double        con,
                        bool is_new_undominated_point, bool /* is_dominated_by_archive */)
                    {
                        // Accept new undominated points.
                        if (is_new_undominated_point)
                            return true;
                        // Constraint domination.
                        if (backup_con < con)
                            return true;
                        else if (backup_con > con)
                            return false;
                        // Linear Scalarization - maximize 'distance' from nadir \w weights
                        double s_backup = 0.0;
                        double s = 0.0;
                        for (int o = 0; o < number_of_objectives; ++o)
                        {
                            s_backup += (backup_obj[o] - nadir[o]) / range[o] * weights[o];
                            s        += (       obj[o] - nadir[o]) / range[o] * weights[o];
                        }

                        return s >= s_backup;
                    },
                        population[p], objective_values[p],  constraint_values[p], solution_metadata[p],
                        population[p], objective_values[p], &constraint_values[p], solution_metadata[p],
                        true, multipass);
                }
                //
                for (int i_w = 0; i_w < population_size; ++i_w)
                    free(all_weights[i_w]);
                free(all_weights);
                free(assignment);
                free(nadir); free(ideal); free(range);
            }
            break;
        case 6:
            {
            double **all_weights                = computeApproximatelyEquidistantPositiveDirections(population_size, number_of_objectives);
            // normalizeDirectionsOrder2(all_weights, population_size, number_of_objectives);
            int     *assignment_point_to_weight = scalarizedAssignWeightVectorsEvenly(number_of_objectives, optimization, population_size, objective_values, population_size, all_weights, range, ideal);
            // int     *assignment_point_to_weight = createRandomOrdering(population_size);
            // int     *assignment_weight_to_point = inversePermutation(assignment_point_to_weight, population_size);
            // char scalarization_kind = 0;

            for (int p = 0; p < population_size; ++p)
            {
                // Apply hillclimber
                performHillClimber([&p, &all_weights, &nadir, &range, &assignment_point_to_weight](
                    char* /* backup */, double* backup_obj, double backup_con,
                    char* /* result */, double*        obj, double        con,
                    bool is_new_undominated_point, bool /* is_dominated_by_archive */)
                {
                    // Accept new undominated points.
                    if (is_new_undominated_point)
                        return true;
                    // Constraint domination.
                    if (backup_con < con)
                        return true;
                    else if (backup_con > con)
                        return false;
                    // int p_other = p;
                    int* ordering = createRandomOrdering(population_size);
                    for (int p_other_idx = 0; p_other_idx < population_size; ++p_other_idx)
                    {
                        int p_other = ordering[p_other_idx];
                        int self_w_idx = assignment_point_to_weight[p];
                        int other_w_idx = assignment_point_to_weight[p_other];

                        double* weights_other = all_weights[other_w_idx];
                        double* weights_self  = all_weights[self_w_idx];

                        double s_other_w_other = 0.0;
                        double s_backup_w_other = 0.0;
                        double s_w_other = 0.0;

                        double s_other_w_self = 0.0;
                        double s_backup_w_self = 0.0;
                        double s_w_self = 0.0;

                        for (int o = 0; o < number_of_objectives; ++o)
                        {   
                            // Scalarize other
                            if (p_other != p)
                            {
                                s_other_w_other += (objective_values[p_other][o] - nadir[o]) / range[o] * weights_other[o];
                                s_other_w_self += (objective_values[p_other][o] - nadir[o]) / range[o] * weights_self[o];
                            } 
                            else
                            {
                                s_other_w_other += (backup_obj[o] - nadir[o]) / range[o] * weights_other[o];
                                s_other_w_self  += (backup_obj[o] - nadir[o]) / range[o] * weights_self[o];
                            }
                            // Scalarize backup
                            s_backup_w_other += (backup_obj[o] - nadir[o]) / range[o] * weights_other[o];
                            s_backup_w_self += (backup_obj[o] - nadir[o]) / range[o] * weights_self[o];

                            s_w_other += (obj[o] - nadir[o]) / range[o] * weights_other[o];
                            s_w_self += (obj[o] - nadir[o]) / range[o] * weights_self[o];
                        }

                        // Reference state: s_other_w_other + s_backup_w_self
                        // Swap-only state: s_other_w_self + s_backup_w_other
                        // Accept-only state: s_other_w_other + s_w_self
                        // Both: s_other_w_self + s_w_other

                        if (s_other_w_self + s_w_other > s_other_w_other + s_backup_w_self)
                        {
                            // Swap weight assignment and accept.
                            assignment_point_to_weight[p_other] = assignment_point_to_weight[p];
                            assignment_point_to_weight[p]       = other_w_idx;                            
                            return true;
                        }
                        else if (s_other_w_self + s_backup_w_other > s_other_w_other + s_backup_w_self)
                        {
                            // Swap weight assignment, but don't accept: search further.
                            assignment_point_to_weight[p_other] = assignment_point_to_weight[p];
                            assignment_point_to_weight[p]       = other_w_idx;  
                        }
                        else if (s_other_w_other + s_w_self > s_other_w_other + s_backup_w_self)
                        {
                            // :)
                            return true;
                        }
                    }
                    free(ordering);

                    return false;
                },
                    population[p], objective_values[p],  constraint_values[p], solution_metadata[p],
                    population[p], objective_values[p], &constraint_values[p], solution_metadata[p],
                    true, multipass);
            }
            //
            for (int i_w = 0; i_w < population_size; ++i_w)
                free(all_weights[i_w]);
            free(all_weights);
            free(assignment_point_to_weight); 
            // free(assignment_weight_to_point);
            free(nadir); free(ideal); free(range);
            }
            break;
    }

    if (write_initial_population)
        writeCurrentPopulation(2); // Write population at initialization 
}
/**
 * Computes the ranges of all fitness values
 * of all solutions currently in the populations.
 */
void computeObjectiveRanges( void )
{
    int    i, j;
    double low, high;

    for( j = 0; j < number_of_objectives; j++ )
    {
        low  = objective_values[0][j];
        high = objective_values[0][j];

        for( i = 0; i < population_size; i++ )
        {
            if( objective_values[i][j] < low )
                low = objective_values[i][j];
            if( objective_values[i][j] > high )
                high = objective_values[i][j];
        }
        
        objective_ranges[j] = high - low;

        if (high - low == 0)
        {
            // std::cout << "Spatial dimension " << j << " has converged to a singlular value " << high << ".\n";
            // std::cout << "This may lead to glitches around evaluation " << number_of_evaluations << "." << std::endl;
            // Fallback to using the archive, maybe a good thing to include in any case.
            //! Shouldn't the copying of extreme solutions ensure this anyways?
            // TODO: This may be indicative of this being out-of-order.
            
            for( i = 0; i < elitist_archive_size; i++ )
            {
                if( elitist_archive_objective_values[i][j] < low )
                    low = elitist_archive_objective_values[i][j];
                if( elitist_archive_objective_values[i][j] > high )
                    high = elitist_archive_objective_values[i][j];
            }

            objective_ranges[j] = high - low;

        }

        if (high - low == 0)
        {
            // This should very much be an edge case, as this would indicate both a converged population
            // and an archive that consists of a single point.
            // This would very much be the case for a single objective problem however -- or a problem that is so agreeable
            // That a single-objective approach would work as well.
            
            // We choose to not normalize in this case.
            // In any case: this spatial dimension should no longer matter for distances.
            objective_ranges[j] = 1;
        }

    }
}

void performClusteringDefault()
{
    int size_of_one_cluster;

    population_indices_of_cluster_members = clustering(objective_values, population_size, number_of_objectives, 
                            number_of_mixing_components, &size_of_one_cluster, objective_means_scaled);
    population_cluster_sizes = (int*) Malloc(number_of_mixing_components*sizeof(int));
    for(int k = 0; k < number_of_mixing_components; k++)    
        population_cluster_sizes[k] = size_of_one_cluster;
}

bool ranks_determined = false;

void determineRanks()
{
    if (ranks_determined) return;
    rank.resize(population_size);
    rank_order.resize(0);
    max_rank = nondominated_sort(population_size, objective_values, rank.data(), rank_order, rank_start_end);
    ranks_determined = true;
}

void computeHVForRank(std::vector<double> &adjusted_nadir, double **&points, int &r)
{
    std::pair<int, int> &start_end = rank_start_end[r];
    int start_idx_rank_order = std::get<0>(start_end);
    int end_idx_rank_order = std::get<1>(start_end);

    int number_of_points_in_rank = end_idx_rank_order - start_idx_rank_order;

    hv_per_rank[r] = compute2DHypervolumeSubsetAndContributions(
        &rank_order[start_idx_rank_order], number_of_points_in_rank, optimization, adjusted_nadir.data(), points,
        hv_contributions_per_population_member.data());
}
void computeHVPerRank()
{
    assert(number_of_objectives == 2);
    hv_per_rank.resize(max_rank);
    hv_contributions_per_population_member.resize(population_size);

    for (int r = 0; r < max_rank; ++r)
    {
        computeHVForRank(adjusted_nadir, objective_values, r);
    }
}

void determineORPs()
{
    int max_rank = 1;

    // Vector of indices that have rank of at most `max_rank`.
    std::vector<int> reference_solutions(population_size);
    setVectorToRandomOrdering(reference_solutions);
    // std::iota(reference_solutions.begin(), reference_solutions.end(), 0);
    auto it_end = std::remove_if(reference_solutions.begin(), reference_solutions.end(),
        [max_rank](int idx){ return rank[idx] >= max_rank; });
    reference_solutions.resize(std::distance(reference_solutions.begin(), it_end));
    
    // 
    objective_ordered_reference_points.resize(number_of_objectives);
    for (int o = 0; o < number_of_objectives; ++o)
    {
        objective_ordered_reference_points[o].resize(reference_solutions.size());
        std::copy(reference_solutions.begin(), reference_solutions.end(), objective_ordered_reference_points[o].begin());
        std::sort(objective_ordered_reference_points[o].begin(), objective_ordered_reference_points[o].end(),
            [o](int a, int b){ return objective_values[a][o] < objective_values[b][o]; });
    }
}

int getReferenceObjectiveRank(int o, std::vector<int> &ordered_reference_points, double value)
{
    size_t lower = 0;
    size_t upper = ordered_reference_points.size();
    size_t mid = (lower + upper) / 2;
    while (mid != lower)
    {
        double mid_value = objective_values[ordered_reference_points[mid]][o];
        
        if (value < mid_value) // Middle is too high
            upper = mid;
        else // Too low
            lower = mid;
        
        mid = (lower + upper) / 2;
    }

    return mid;
}

void getSuggestedPoint(double* point_in, std::vector<double> &suggested_point)
{
    bool all_nadir = true;
    for (int o = 0; o < number_of_objectives; ++o)
    {
        std::vector<int> &ordered_reference_points = objective_ordered_reference_points[o];
        int rank = getReferenceObjectiveRank(o, ordered_reference_points, point_in[o]);
        if (rank == 0)
            suggested_point[o] = nadir_point[o]; // Worst at an objective uses the nadir point.
        else if (rank == static_cast<int>(ordered_reference_points.size() - 1))
            { suggested_point[o] = utopian_point[o]; all_nadir = false; } // Being the best for an objective
        else
            { suggested_point[o] = (objective_values[rank - 1][o] + objective_values[rank + 1][o]) / 2; all_nadir = false; }
    }
    
    // Point is worse than all reference points. Set to itself as fallback.
    if (all_nadir)
    {
        for (int o = 0; o < number_of_objectives; ++o)
        {
            suggested_point[o] = point_in[o];
        }
    }
}

void insertExtremeArchiveSolution()
{
    if (!preserve_extreme_frontal_points)
        return;
    
    int* randomOrder;
    std::vector<int> archiveExtremeIndices(number_of_objectives);
    for (int o = 0; o < number_of_objectives; ++o)
    {
        randomOrder = createRandomOrdering(elitist_archive_size);
        int currentExtremeIndex = 0;
        double currentExtremeObjectiveValue = (optimization[o] == MAXIMIZATION) ? 
            -INFINITY /* MAX */ : INFINITY /* MIN */ ;
        for (int eoi = 0; eoi < elitist_archive_size; ++eoi)
        {
            int i = randomOrder[eoi];
            double v = elitist_archive_objective_values[i][o];
            double dir = ( (optimization[o] == MAXIMIZATION) ? 1.0 : -1.0 );
            if (v * dir > currentExtremeObjectiveValue)
            {
                currentExtremeIndex = i;
                currentExtremeObjectiveValue = v;
            }
        }

        // std::cout << "Extreme for objective " << o << " in archive is " << currentExtremeObjectiveValue << ".\n"; 

        bool any_matching_extreme_value_for_objective_in_population = false;
        // Check if exists in population.
        // for (int i = 0; i < population_size; ++i)
        // {
        //     if (objective_values[i][o] == currentExtremeObjectiveValue)
        //     {
        //         // std::cout << "Found solution with same objective as extreme.\n"; 
        //         bool all_equal = true;
        //         for (int o2 = 0; o2 < number_of_objectives; ++o2)
        //         {
        //             if (objective_values[i][o2] != elitist_archive_objective_values[currentExtremeIndex][o2])
        //             {
        //                 // std::cout << "But not all equal...\n"; 
        //                 all_equal = false;
        //                 break;
        //             }
        //         }
        //         if (! all_equal) continue;
        //         // std::cout << "And it was identical\n"; 

        //         any_matching_extreme_value_for_objective_in_population = true;
        //         break;
        //     }
        // }

        if (any_matching_extreme_value_for_objective_in_population)
        {
            archiveExtremeIndices[o] = -1;
        }
        else
        {
            archiveExtremeIndices[o] = currentExtremeIndex;
        }

        free(randomOrder);
    }

    randomOrder = createRandomOrdering(number_of_objectives);
    // Variable is only used for rank based replacement.
    // int offset = 1;

    for (int oi = 0; oi < number_of_objectives; ++oi)
    {
        int o = randomOrder[oi];
        int a = archiveExtremeIndices[o];
        // Negative: population already matches extreme value for objective.
        if (a < 0) continue;
        
        int i;
        // Rank based replacement
        // Note: solution-being-replaced may be far away, but is based on rank.
        // i = rank_order[population_size - offset];
        // ++offset;

        // Nearest replacement
        i = 0;
        double di = INFINITY;
        for (int potential_i = 0; potential_i < population_size; ++potential_i)
        {
            double d = distanceEuclidean(objective_values[potential_i], elitist_archive_objective_values[a], number_of_objectives, objective_ranges);
            if (d < di)
            {
                i = potential_i;
                di = d;
            }
        }

        // std::cout << "Replaced population member " << i << "(";
        // for (int v = 0; v < number_of_objectives; ++v)
        // {
        //     if (v != 0)
        //         std::cout << ",";
        //     std::cout << objective_values[i][v];
        // }
        // std::cout << ") with elitist for objective " << o << " namely " << a << " (";
        // for (int v = 0; v < number_of_objectives; ++v)
        // {
        //     if (v != 0)
        //         std::cout << ",";
        //     std::cout << elitist_archive_objective_values[a][v];
        // }
        // std::cout << ")." << std::endl; 

        copyFromAToB(elitist_archive[a], elitist_archive_objective_values[a], elitist_archive_constraint_values[a], elitist_archive_solution_metadata[a],  /* A */
                           population[i],                 objective_values[i],                &constraint_values[i], solution_metadata[i]); /* B */
    }
    free(randomOrder);
    
}

void determineLines()
{
    // Helper for normalization (on / off)
    double* ranges = NULL;
    if (normalize_objectives_for_lines)
        ranges = objective_ranges;

    // Determine Nadir & Ideal
    std::vector<double> nadir(number_of_objectives);
    std::vector<double> ideal(number_of_objectives);
    for (int o = 0; o < number_of_objectives; ++o)
    {
        nadir[o] = (optimization[o] == MAXIMIZATION) ?  INFINITY : -INFINITY;
        ideal[o] = (optimization[o] == MAXIMIZATION) ? -INFINITY :  INFINITY;
    }
    for (int p = 0; p < population_size; ++p)
    {
        for (int o = 0; o < number_of_objectives; ++o)
        {
            nadir[o] = (optimization[o] == MAXIMIZATION) ? 
                std::min(nadir[o], objective_values[p][o]) : 
                std::max(nadir[o], objective_values[p][o]);
            ideal[o] = (optimization[o] == MAXIMIZATION) ? 
                std::max(ideal[o], objective_values[p][o]) : 
                std::min(ideal[o], objective_values[p][o]);
        }
    }

    int* selection_best = NULL;
    int* selection_worst = NULL;
    if (line_mode == 0 || line_mode == 1 || line_mode == 2)
    {
        // A-A: Determine ranks of solutions
        // Note to self: nondominated_sort uses the domination check function, which directly uses the global variable number_of_objectives.
        // Another note to self: this is now an assumption to have been ran before determineLines!
        
        // A-B: Determine best_rank_max and worst_rank_min.
        
        // Minimum 'percentage' of samples in max and min.
        double p = std::max(0.05, 2 * ((double) number_of_mixing_components / (double) population_size));

        std::vector<int> rank_counts(max_rank);
        for (int r: rank) rank_counts[r] += 1;

        int best_rank_min = 0;
        int best_rank_max = max_rank; // TBD
        int worst_rank_min = 0; // TBD
        int worst_rank_max = max_rank;

        int samples_accounted_for = 0;
        int samples_needed_best_max = std::ceil(population_size * p);
        int samples_needed_worst_min = std::floor(population_size * (1 - p));
        for (int r = 0; r < max_rank; ++r)
        {
            int current_rank_count = rank_counts[r];
            // If previously we did not hit the limit, but now we do.
            // The current rank is the one we want.
            if (samples_accounted_for < samples_needed_best_max &&
                samples_accounted_for + current_rank_count >= samples_needed_best_max) 
                best_rank_max = r;
            // Similarily, for worst_min
            if (samples_accounted_for < samples_needed_worst_min &&
                samples_accounted_for + current_rank_count >= samples_needed_worst_min) 
                worst_rank_min = r;
            samples_accounted_for += current_rank_count;
        }

        // A-C: Filter into indices based on ranks.
        std::vector<int> indices_best; indices_best.reserve(population_size);
        std::vector<int> indices_worst; indices_worst.reserve(population_size);
        for (int i = 0; i < population_size; ++i)
        {
            int i_rank = rank[i];
            if (i_rank >= best_rank_min && i_rank <= best_rank_max) indices_best.push_back(i);
            if (i_rank >= worst_rank_min && i_rank <= worst_rank_max) indices_worst.push_back(i);
        }

        // A-D: Perform Greedy Subset Scattering - restricted to both subsets.
        selection_best = greedyScatteredSubSubsetSelection(indices_best, objective_values, number_of_objectives, number_of_mixing_components, ranges);
        selection_worst = greedyScatteredSubSubsetSelection(indices_worst, objective_values, number_of_objectives, number_of_mixing_components, ranges);
    }
    else if (line_mode == 3)
    {
        // A-A: Determine unit-normal vectors of population.
        std::vector<double> normalized_coordinates(population_size * number_of_objectives);
        for (int p = 0; p < population_size; ++p)
        {
            // Determine length (in normalized space)
            double length = 0.0;
            for (int d = 0; d < number_of_objectives; ++d)
            {
                double delta = (objective_values[p][d] - nadir[d]) / ranges[d];
                length += delta * delta;
            }
            length = std::sqrt(length);
            // Store normalized vector
            for (int d = 0; d < number_of_objectives; ++d)
            {
                normalized_coordinates[p * number_of_objectives + d] = (objective_values[p][d] - nadir[d]) / (ranges[d] * length);
            }
        }
        //     Create a vector of pointers (such that greedyScatteredSubsetSelection can be used)
        std::vector<double*> normalized_points(population_size);
        for (int p = 0; p < population_size; ++p)
        {
            normalized_points[p] = &normalized_coordinates[p * number_of_objectives];
        }

        // A-B: Perform greedy scattered subset selection.
        selection_best = greedyScatteredSubsetSelection(normalized_points.data(), population_size, number_of_objectives, number_of_mixing_components);
        // Note: This shouldn't be needed as it is getting overridden
        //       by the nadir, but assumptions are annoying (and the copying)
        //       assumes we have all the points we need, not less (e.g. 0).
        selection_worst = greedyScatteredSubsetSelection(normalized_points.data(), population_size, number_of_objectives, number_of_mixing_components);
    }
    else
    {
        std::cerr << "Unknown line mode." << std::endl;
        std::exit(1);
    }
    
    // D (extra): Copy selection
    
    std::vector<std::vector<double>> selection_best_objective_values(number_of_mixing_components);
    std::vector<std::vector<double>> selection_worst_objective_values(number_of_mixing_components);
    for (int c = 0; c < number_of_mixing_components; ++c)
    {
        double* c_best_objective_values = objective_values[selection_best[c]];
        selection_best_objective_values[c] = std::vector<double>(c_best_objective_values, c_best_objective_values + number_of_objectives); 
        double* c_worst_objective_values = objective_values[selection_worst[c]];
        selection_worst_objective_values[c] = std::vector<double>(c_worst_objective_values, c_worst_objective_values + number_of_objectives);
    }

    // E: Apply transformations (where applicable)

    // Damping
    if (damping_factor > 0.0 && (static_cast<int>(lines.size()) == number_of_mixing_components))
    {
        // Compute distances
        std::vector<double> distance_prev_best_to_current_best(number_of_mixing_components * number_of_mixing_components);
        std::vector<double> distance_prev_worst_to_current_worst(number_of_mixing_components * number_of_mixing_components);

        for (int idx_current = 0; idx_current < number_of_mixing_components; ++idx_current)
        {
            for (int idx_prev = 0; idx_prev < number_of_mixing_components; ++idx_prev)
            {
                double d_best = distanceEuclidean(selection_best_objective_values[idx_current].data(),
                                            lines[idx_prev].first.data(),
                                            number_of_objectives, ranges);
                double d_worst = distanceEuclidean(selection_worst_objective_values[idx_current].data(),
                                            lines[idx_prev].second.data(),
                                            number_of_objectives, ranges);
                distance_prev_best_to_current_best[idx_current * number_of_mixing_components + idx_prev] = d_best;
                distance_prev_worst_to_current_worst[idx_current * number_of_mixing_components + idx_prev] = d_worst;
            }
        }
        
        // Solve assignment problem
        // ?: Maybe use greedy algorithm instead? May be faster if this turns out to be a bottleneck.
        std::vector<int64_t> assignment_bests(number_of_mixing_components);
        int success_best = solve_rectangular_linear_sum_assignment(number_of_mixing_components, number_of_mixing_components, distance_prev_best_to_current_best.data(), assignment_bests.data());
        assert(success_best == 0);

        std::vector<int64_t> assignment_worsts(number_of_mixing_components);
        int success_worst = solve_rectangular_linear_sum_assignment(number_of_mixing_components, number_of_mixing_components, distance_prev_worst_to_current_worst.data(), assignment_worsts.data());
        assert(success_worst == 0);

        // Apply damping through an averaging procedure
        for (int i = 0; i < number_of_mixing_components; ++i)
        {
            for(int o = 0; o < number_of_objectives; ++o)
            {
                selection_best_objective_values[i][o] *= 1 - damping_factor;
                selection_best_objective_values[i][o] += damping_factor * lines[assignment_bests[i]].first[o];
                
                selection_worst_objective_values[i][o] *= 1 - damping_factor;
                selection_worst_objective_values[i][o] += damping_factor * lines[assignment_worsts[i]].second[o];
            }
        }
    }

    // Force axis-alignment
    if (align_nearest_lines_to_axis)
    {
        // Nadir, with single subsitution - for use as replacement point.
        std::vector<double> nadir_s(nadir); 
        // For each axis, try overriding (but in a random order, to avoid biases in cases of a tie)
        int* objectives_ord = createRandomOrdering(number_of_objectives);
        for (int o_idx = 0; o_idx < number_of_objectives; ++o_idx)
        {
            int o = objectives_ord[o_idx];
            // Subsitute one part for the ideal point.
            nadir_s[o] = ideal[o];
            int nearest = -1;
            double nearest_distance = INFINITY;
            for (int i = 0; i < number_of_mixing_components; ++i )
            {
                int idx = selection_best[i];
                // Already replaced, skip
                if (idx < 0) continue;
                double distance = distanceEuclidean(selection_best_objective_values[i].data(), nadir_s.data(), number_of_objectives, ranges);
                if (distance < nearest_distance)
                {
                    nearest = i;
                    nearest_distance = distance;
                }
            }
            // Replace if any found
            if (nearest >= 0)
            {
                // Replace
                std::copy(nadir_s.begin(), nadir_s.end(), selection_best_objective_values[nearest].begin());
                // Mark as replaced
                selection_best[nearest] = -o - 1;
            }
            // Reset nadir_s to nadir.
            nadir_s[o] = nadir[o];
        }
        free(objectives_ord);
    }

    // F: Determine assignment from worst to best
    //    Note: This is unnecessary when one of the points is identical
    //          (any assignment would work / no assignment is neccesary)
    // Note: assignment is from best (ordered 0..n-1) to worst (order given by assignment)
    std::vector<int64_t> assignment(number_of_mixing_components);
    if (line_mode == 0 || line_mode == 1)
    {
        // F1: Calculate the distance matrix to perform matching on.
        std::vector<double> worst_to_best_distances(number_of_mixing_components * number_of_mixing_components);
        for (int idx_best = 0; idx_best < number_of_mixing_components; ++idx_best)
        {
            for (int idx_worst = 0; idx_worst < number_of_mixing_components; ++idx_worst)
            {
                double d = distanceEuclidean(selection_best_objective_values[idx_best].data(),
                                            selection_worst_objective_values[idx_worst].data(),
                                            number_of_objectives, ranges);
                worst_to_best_distances[idx_best * number_of_mixing_components + idx_worst] = d;
            }
        }

        // F2: Solve matching problem - Uses the implementation from scipy.

        int success = solve_rectangular_linear_sum_assignment(number_of_mixing_components, number_of_mixing_components, worst_to_best_distances.data(), assignment.data());
        // As all distances are finite, success should be guaranteed!
        assert(success == 0);
    }

    // G: Store the resulting lines.
    lines.clear();
    lines.resize(number_of_mixing_components);
    lines_direction_normalized.clear();
    lines_direction_normalized.resize(number_of_mixing_components);

    for (int l = 0; l < number_of_mixing_components; ++l)
    {        
        double *o_best = selection_best_objective_values[l].data();
        double *o_worst;

        if (line_mode == 2 || line_mode == 3)
        {
            o_worst = nadir.data();
        }
        else
        {
            o_worst = selection_worst_objective_values[assignment[l]].data();
        }

        // Compute normalized vector. And store that as well.
        // Important for fast projections!
        std::vector<double> normalized_difference(number_of_objectives);
        double length = 0.0;
        for (int o = 0; o < number_of_objectives; ++o)
        {
            double delta = o_best[o] - o_worst[o];
            // Note: normalize individual dimensions of the vector when using objective normalization as well.
            if (ranges != NULL)
                delta /= ranges[o];
            normalized_difference[o] = delta;

            length += delta*delta;
        }
        length = std::sqrt(length);

        // Note: if length is zero, skipping this makes normalized_difference be the zero vector
        // And results in a computation that makes distance behave like a point.
        if (length > 0)
        {
            for (int o = 0; o < number_of_objectives; ++o)
            {
                normalized_difference[o] = normalized_difference[o] / length;
            }
        }
        else if (line_mode == 1)
        {
            // Use nadir as worst instead.
            o_worst = nadir.data();
            length = 0.0;
            for (int o = 0; o < number_of_objectives; ++o)
            {
                double delta = o_best[o] - o_worst[o];
                // Note: normalize individual dimensions of the vector when using objective normalization as well.
                if (ranges != NULL)
                    delta /= ranges[o];
                normalized_difference[o] = delta;

                length += delta*delta;
            }
            length = std::sqrt(length);
            if (length > 0)
            {
                for (int o = 0; o < number_of_objectives; ++o)
                {
                    normalized_difference[o] = normalized_difference[o] / length;
                }
            }
        }
        
        // Note: clone so the lines don't change!
        // ?: Reuse previous copy here by using std::move from selection_best_objective_values & selection_worst_objective_values?
        // !: Will fail when nadir replacement has been performed.
        lines[l] = (std::pair(
            std::vector<double>(o_best, o_best + number_of_objectives),
            std::vector<double>(o_worst, o_worst + number_of_objectives))
        );

        lines_direction_normalized[l] = std::move(normalized_difference);
    }

    if ( selection_best != NULL) free(selection_best);
    if (selection_worst != NULL) free(selection_worst);
}

/**
 * Compute the euclidean distance to a line
 **/
double euclideanDistanceToLine(double *a, double *u, double *p, int number_of_dimensions, double* ranges = NULL)
{
    double dot_product = 0.0;
    for (int d = 0; d < number_of_dimensions; ++d)
    {
        double p_min_a_d = p[d] - a[d];
        if (ranges != NULL)
            p_min_a_d /= ranges[d];
        dot_product += p_min_a_d * u[d];
    }
    double result = 0.0;
    for (int d = 0; d < number_of_dimensions; ++d)
    {
        double p_min_a_d = p[d] - a[d];
        if (ranges != NULL)
            p_min_a_d /= ranges[d];
        double v_d = p_min_a_d - (dot_product) * u[d];
        double v_d_sq = v_d * v_d;
        result += v_d_sq;
    }
    return std::sqrt(result);
}

double computeLinearScalarization(double *a, double *u, double *p, int number_of_dimensions, double* ranges = NULL)
{
    double dot_product = 0.0;
    for (int d = 0; d < number_of_dimensions; ++d)
    {
        double p_min_a_d = p[d] - a[d];
        if (ranges != NULL)
            p_min_a_d /= ranges[d];
        dot_product += p_min_a_d * u[d];
    }
    return dot_product;
}

/**
 * Compute the euclidean distance to a ray, starting at a.
 **/
double euclideanDistanceToRay(double *a, double *u, double *p, int number_of_dimensions, double* ranges = NULL)
{
    double dot_product = 0.0;
    for (int d = 0; d < number_of_dimensions; ++d)
    {
        double p_min_a_d = p[d] - a[d];
        if (ranges != NULL)
            p_min_a_d /= ranges[d];
        dot_product += p_min_a_d * u[d];
    }
    dot_product = std::max(0.0, dot_product);
    double result = 0.0;
    for (int d = 0; d < number_of_dimensions; ++d)
    {
        double p_min_a_d = p[d] - a[d];
        if (ranges != NULL)
            p_min_a_d /= ranges[d];
        double v_d = p_min_a_d - (dot_product) * u[d];
        double v_d_sq = v_d * v_d;
        result += v_d_sq;
    }
    return std::sqrt(result);
}

void computeNearestLineForEachIndividual()
{
    // nearest_line
    nearest_line.resize(population_size);
    for (int p = 0; p < population_size; ++p)
    {
        int p_nearest_line = 0;
        double p_nearest_line_distance = INFINITY;
        for (size_t line_idx = 0; line_idx < lines.size(); ++line_idx)
        {
            auto current_line = lines[line_idx];
            double distance = euclideanDistanceToRay(current_line.first.data(), lines_direction_normalized[line_idx].data(), objective_values[p], number_of_objectives);
            if (distance < p_nearest_line_distance)
            {
                p_nearest_line = line_idx;
                p_nearest_line_distance = distance;
            }
        }
        nearest_line[p] = p_nearest_line;
    }
}

void performClusteringLine()
{
    // Note: this method assumes that determineLines & computeNearestLineForEachIndividual has been called.
    // It should convert nearest_line to a cluster assignment.
    assert(number_of_mixing_components == (int) lines.size());

    // Create vectors of membership
    std::vector<std::vector<int>> cluster_memberships(number_of_mixing_components);
    for (int i = 0; i < population_size; ++i)
    {
        cluster_memberships[nearest_line[i]].push_back(i);
    }

    // Assign clusters.
    population_cluster_sizes = (int*) Malloc(population_size * sizeof(int));
    population_indices_of_cluster_members = (int**) Malloc(sizeof(int*) * number_of_mixing_components);
    for (int c_idx = 0; c_idx < number_of_mixing_components; ++c_idx)
    {
        int s = (int) cluster_memberships[c_idx].size();
        // C-ify.
        population_indices_of_cluster_members[c_idx] = (int*) Malloc(sizeof(int) * s);
        // And compute means.
        objective_means_scaled[c_idx] = (double*) Malloc(sizeof(double) * number_of_objectives);
        for (int o = 0; o < number_of_objectives; ++o)
        {
            objective_means_scaled[c_idx][o] = 0.0;
        }
        for (int i = 0; i < s; ++i)
        {
            int p_idx = cluster_memberships[c_idx][i];
            population_indices_of_cluster_members[c_idx][i] = p_idx;
            for (int o = 0; o < number_of_objectives; ++o)
            {
                objective_means_scaled[c_idx][o] += objective_values[p_idx][o] / objective_ranges[o];
            }
        }
        for (int o = 0; o < number_of_objectives; ++o)
        {
            objective_means_scaled[c_idx][o] += objective_means_scaled[c_idx][o] / ((double) s);
        }
        population_cluster_sizes[c_idx] = s;
    }

}

// Like performClusteringLine, but selects the k-Nearest Neighbors of a line instead.
void performClusteringLineKNN()
{
    // Note: this method assumes that determineLines & computeNearestLineForEachIndividual has been called.
    // It should convert nearest_line to a cluster assignment.
    assert(number_of_mixing_components == (int) lines.size());

    // Note: About a factor of two overlap.
    int k = std::min(population_size, (int) std::ceil(population_size / number_of_mixing_components * 2));

    // Create vectors of membership
    std::vector<std::vector<int>> cluster_memberships(number_of_mixing_components);
    std::vector<std::pair<double, int>> distance_idx_pairs(population_size);

    // Keep track of whether a every item has a cluster, such that we can add it to the nearest cluster if it is not contained
    // in any.
    std::vector<bool> has_cluster(population_size);
    std::fill(has_cluster.begin(), has_cluster.end(), false);

    // Determine Clusters
    for (int line_idx = 0; line_idx < number_of_mixing_components; ++line_idx)
    {
        auto current_line = lines[line_idx];
        // Fill vector
        for (int i = 0; i < population_size; ++i)
        {
            double distance = euclideanDistanceToRay(current_line.first.data(), lines_direction_normalized[line_idx].data(), objective_values[i], number_of_objectives);
            distance_idx_pairs[i] = std::pair(distance, i);
        }
        // Pseudo-sort the vector in place so that the kth element is sorted properly with respect to all other items.
        std::nth_element(distance_idx_pairs.begin(), distance_idx_pairs.begin() + k, distance_idx_pairs.end());
        // assert(distance_idx_pairs[k - 1] <= distance_idx_pairs[k]);
        // assert(distance_idx_pairs[k] <= distance_idx_pairs[k + 1]);

        // Fill cluster
        for (int idx = 0; idx <= k; ++idx)
        {
            int i = distance_idx_pairs[idx].second;
            cluster_memberships[line_idx].push_back(i);
            has_cluster[i] = true;
        }
    }

    // Any population members get added to the nearest cluster using nearest line.
    for (int i = 0; i < population_size; ++i)
    {
        if (!has_cluster[i])
            cluster_memberships[nearest_line[i]].push_back(i);
    }

    // Assign clusters.
    population_cluster_sizes = (int*) Malloc(population_size * sizeof(int));
    population_indices_of_cluster_members = (int**) Malloc(sizeof(int*) * number_of_mixing_components);
    for (int c_idx = 0; c_idx < number_of_mixing_components; ++c_idx)
    {
        int s = (int) cluster_memberships[c_idx].size();
        // C-ify.
        population_indices_of_cluster_members[c_idx] = (int*) Malloc(sizeof(int) * s);
        // And compute means.
        objective_means_scaled[c_idx] = (double*) Malloc(sizeof(double) * number_of_objectives);
        for (int o = 0; o < number_of_objectives; ++o)
        {
            objective_means_scaled[c_idx][o] = 0.0;
        }
        for (int i = 0; i < s; ++i)
        {
            int p_idx = cluster_memberships[c_idx][i];
            population_indices_of_cluster_members[c_idx][i] = p_idx;
            for (int o = 0; o < number_of_objectives; ++o)
            {
                objective_means_scaled[c_idx][o] += objective_values[p_idx][o] / objective_ranges[o];
            }
        }
        for (int o = 0; o < number_of_objectives; ++o)
        {
            objective_means_scaled[c_idx][o] += objective_means_scaled[c_idx][o] / ((double) s);
        }
        population_cluster_sizes[c_idx] = s;
    }
}

void assignSolutionsToLinesByDistanceEvenly()
{
    // First: determine the number of repetitions required
    // such that all points can be assigned to a line.
    int number_of_lines = lines.size();
    int line_repetitions = std::ceil(static_cast<double>(population_size) / static_cast<double>(number_of_lines));
    int number_of_lines_with_repetitions = number_of_lines * line_repetitions;

    // Construct the distance matrix.
    // Row major: population index -by- line with repetitions.
    std::vector<double> distance_matrix(population_size * number_of_lines_with_repetitions);

    for (int p_idx = 0; p_idx < population_size; ++p_idx)
    {
        for (int l_idx = 0; l_idx < number_of_lines; ++l_idx)
        {
            // Determine distance
            double distance = euclideanDistanceToRay(nadir_point.data(), lines_direction_normalized[l_idx].data(), objective_values[p_idx], number_of_objectives, objective_ranges);
            
            // Determine area to fill
            int idx = p_idx * number_of_lines * line_repetitions + l_idx * line_repetitions;
            auto rep_region_start = distance_matrix.begin() + idx;
            auto rep_region_end   = rep_region_start + line_repetitions;
            
            // Fill!
            std::fill(rep_region_start, rep_region_end, distance);
        }
    }
    // Compute assignment
    std::vector<long> assignment(population_size);
    solve_rectangular_linear_sum_assignment(population_size, number_of_lines_with_repetitions, distance_matrix.data(), assignment.data());

    // Set assigned line (transform duplicates back to their original line as well)
    assigned_line.resize(population_size);
    std::transform(assignment.begin(), assignment.end(), assigned_line.begin(), [line_repetitions](long t) { return t / line_repetitions; } );

    //! DEBUG
    if (log_agreements_with_nearest_when_assigning_lines)
    {
        int count_agreements = 0;
        for (int i = 0; i < population_size; ++i)
        {
            count_agreements += assigned_line[i] == nearest_line[i];
        }
        std::cout << "Assigned individuals to lines, " << count_agreements << " out of " << population_size << " were assigned to their nearest line." << std::endl; 
    }
}

void assignSolutionsToLines()
{
    switch (line_assignment_mode)
    {
        case 0:
        {
            assigned_line = nearest_line;
        }
        break;
        case 1:
        {
            assignSolutionsToLinesByDistanceEvenly();
        }
        break;
    }
}

/**
 * Perform Greedy Subset Scattering to obtain a set of clusters.
 *
 * @param number_of_points      How many points are there to be clustered
 * @param number_of_clusters    How many clusters should be created
 * @param distance              Function accepting two indices, returning the distance between the two points.
 * @param indices_left          Which indices to cluster
 */
template <typename T>
void performGSSClustering(int number_of_points, int number_of_clusters, T&& distance, std::vector<int> &indices_left, std::vector<int> &nearest_leader)
{
    int current_leader_indices_idx;
    int current_leader;
    // Initialize indices
    indices_left.resize(number_of_points);
    nearest_leader.resize(number_of_points);
    std::vector<double> nearest_leader_distance(number_of_points);
    std::fill(nearest_leader_distance.begin(), nearest_leader_distance.end(), INFINITY);

    int number_of_leaders = 0;
    // std::vector<int> cluster_leaders(number_of_clusters);
    // std::iota(indices_left.begin(), indices_left.end(), 0);
    setVectorToRandomOrdering(indices_left);

    // Run an initial iteration without updates to get a good starting point from a random point.
    current_leader_indices_idx = randomInt(number_of_points);
    current_leader = indices_left[current_leader_indices_idx];
    int next_leader = current_leader;
    int next_leader_idx = 0;
    double next_distance = -INFINITY;

    for (size_t i_indices_left = 0; i_indices_left < indices_left.size(); ++i_indices_left)
    {
        int i = indices_left[i_indices_left];
        double distance_i_to_current_leader = distance(current_leader, i);
    
        // Is the new distance the farthest away so far?
        if (distance_i_to_current_leader < next_distance)
            continue;
        
        next_leader = i;
        next_leader_idx = i_indices_left;
        next_distance = distance_i_to_current_leader;
    }
    // Switch to the next leader.
    current_leader = next_leader;
    current_leader_indices_idx = next_leader_idx;
    // Remove next leader from indices left.
    size_t last_indices_left = indices_left.size() - 1;
    std::swap(indices_left[current_leader_indices_idx], indices_left[last_indices_left]);
    indices_left.resize(last_indices_left);
    // Add to cluster leaders
    // cluster_leaders.push_back(current_leader);
    number_of_leaders += 1;
    // Set nearest for current leader
    nearest_leader[current_leader] = number_of_leaders - 1; 
    
    while (number_of_leaders < number_of_clusters)
    {
        next_leader = current_leader;
        next_leader_idx = 0;
        next_distance = -INFINITY;

        for (size_t i_indices_left = 0; i_indices_left < indices_left.size(); ++i_indices_left)
        {
            int i = indices_left[i_indices_left];
            double distance_i_to_current_leader = distance(current_leader, i);

            // Is the new distance better (and do we update the distance-to-nearest-list)
            if (distance_i_to_current_leader > nearest_leader_distance[i])
                continue;
            
            nearest_leader_distance[i] = distance_i_to_current_leader;
            // nearest_leader[i] = current_leader; // Identify cluster by leader index.
            nearest_leader[i] = number_of_leaders - 1; // Identify cluster by index in the order of which the leaders were assigned.
        
            // Is the new distance the farthest away so far?
            if (distance_i_to_current_leader < next_distance)
                continue;
            
            next_leader = i;
            next_leader_idx = i_indices_left;
            next_distance = distance_i_to_current_leader;
        }
        // Switch to the next leader.
        current_leader = next_leader;
        current_leader_indices_idx = next_leader_idx;
        // Remove next leader from indices left.
        size_t last_indices_left = indices_left.size() - 1;
        std::swap(indices_left[current_leader_indices_idx], indices_left[last_indices_left]);
        indices_left.resize(last_indices_left);
        // Add to cluster leaders
        // cluster_leaders.push_back(current_leader);
        number_of_leaders += 1;
        // Set cluster of new leader
        nearest_leader[current_leader] = number_of_leaders - 1; 
    }
}

/**
 * Perform Greedy Subset Scattering to obtain a set of clusters -- but each sample is present in more clusters. 
 *
 * @param number_of_points                  How many points are there to be clustered
 * @param number_of_clusters                How many clusters should be created
 * @param number_of_assignments_per_point   Number of clusters a point should be assigned to.
 * @param distance                          Function accepting two indices, returning the distance between the two points.
 * @param indices_left                      Which indices to cluster
 */
template <typename T>
void performMultiGSSClustering(int number_of_points, int number_of_clusters, int number_of_assignments_per_point, T&& distance, std::vector<int> &indices_left, std::vector<int> &nearest_leader)
{
    int current_leader_indices_idx;
    int current_leader;
    // Initialize indices
    indices_left.resize(number_of_points);
    nearest_leader.resize(number_of_points * number_of_assignments_per_point);
    std::fill(nearest_leader.begin(), nearest_leader.end(), -1);
    std::vector<double> nearest_leader_distance(number_of_points * number_of_assignments_per_point);
    std::fill(nearest_leader_distance.begin(), nearest_leader_distance.end(), INFINITY);

    if (number_of_clusters > number_of_points)
    {
        number_of_clusters = number_of_points;
    }

    int number_of_leaders = 0;
    // std::vector<int> cluster_leaders(number_of_clusters);
    // std::iota(indices_left.begin(), indices_left.end(), 0);
    setVectorToRandomOrdering(indices_left);

    // Run an initial iteration without updates to get a good starting point from a random point.
    current_leader_indices_idx = randomInt(number_of_points);
    current_leader = indices_left[current_leader_indices_idx];
    int next_leader = current_leader;
    int next_leader_idx = 0;
    double next_distance = -INFINITY;

    for (size_t i_indices_left = 0; i_indices_left < indices_left.size(); ++i_indices_left)
    {
        int i = indices_left[i_indices_left];
        double distance_i_to_current_leader = distance(current_leader, i);
    
        // Is the new distance the farthest away so far?
        if (distance_i_to_current_leader < next_distance)
            continue;
        
        next_leader = i;
        next_leader_idx = i_indices_left;
        next_distance = distance_i_to_current_leader;
    }
    // Switch to the next leader.
    current_leader = next_leader;
    current_leader_indices_idx = next_leader_idx;
    // Remove next leader from indices left.
    size_t last_indices_left = indices_left.size() - 1;
    std::swap(indices_left[current_leader_indices_idx], indices_left[last_indices_left]);
    indices_left.resize(last_indices_left);
    // Add to cluster leaders
    // cluster_leaders.push_back(current_leader);
    number_of_leaders += 1;
    // Set nearest for current leader
    nearest_leader[current_leader] = number_of_leaders - 1; 
    
    while (number_of_leaders < number_of_clusters)
    {
        int current_number_of_assignments = std::min(number_of_leaders, number_of_assignments_per_point);
        // int new_number_of_assignments = std::min(number_of_leaders + 1, number_of_assignments_per_point);
        next_leader = current_leader;
        next_leader_idx = 0;
        next_distance = -INFINITY;

        for (size_t i_indices_left = 0; i_indices_left < indices_left.size(); ++i_indices_left)
        {
            int i = indices_left[i_indices_left];
            double distance_i_to_current_leader = distance(current_leader, i);

            int i_last = i + number_of_points * (current_number_of_assignments - 1);
            // Is the new distance better (and do we update the distance-to-nearest-list)
            // Last item is defined to be the furthest away
            if (number_of_leaders != number_of_assignments_per_point && 
                distance_i_to_current_leader > nearest_leader_distance[i_last])
                continue;
            
            nearest_leader_distance[i_last] = distance_i_to_current_leader;
            // nearest_leader[i_last] = current_leader; // Identify cluster by leader index.
            nearest_leader[i_last] = number_of_leaders - 1; // Identify cluster by index in the order of which the leaders were assigned.
            // Place new assignment in the correct position (by bubbling)
            for (int o = 0; o < current_number_of_assignments - 1; ++o)
            {
                int i_new_current = i_last - o * number_of_points;
                int i_old_current = i_last - (o + 1) * number_of_points;
                if ( nearest_leader_distance[i_old_current] > nearest_leader_distance[i_new_current] )
                {
                    // Old is farther away than new, swap.
                    std::swap(nearest_leader[i_new_current], nearest_leader[i_old_current]);
                    std::swap(nearest_leader_distance[i_new_current], nearest_leader_distance[i_old_current]);
                }
                else break;
            }

            double new_leader_distance = nearest_leader_distance[i];
            // Is the new leader distance for this solution also the farthest so far?
            // If yes: this soltuion will potentially become the next leader.
            if (new_leader_distance < next_distance)
                continue;
            
            next_leader = i;
            next_leader_idx = i_indices_left;
            next_distance = new_leader_distance;
        }
        // Switch to the next leader.
        current_leader = next_leader;
        current_leader_indices_idx = next_leader_idx;
        // Remove next leader from indices left.
        size_t last_indices_left = indices_left.size() - 1;
        std::swap(indices_left[current_leader_indices_idx], indices_left[last_indices_left]);
        indices_left.resize(last_indices_left);
        // Add to cluster leaders
        // cluster_leaders.push_back(current_leader);
        number_of_leaders += 1;
        // Set cluster of new leader
        nearest_leader[current_leader] = number_of_leaders - 1;
        nearest_leader_distance[current_leader] = 0.0;
    }
}

/**
 * Perform Greedy Subset Scattering to obtain a set of clusters.
 *
 * @param number_of_points      How many points are there to be clustered
 * @param number_of_clusters    How many clusters should be created
 * @param distance              Function accepting two indices, returning the distance between the two points.
 * @param indices_left          Which indices to cluster
 */
template <typename T>
void performGreedySubsetScatteringWithDistance(int number_of_points, int number_of_clusters, T&& distance, std::vector<int> &cluster_leaders)
{
    int current_leader_indices_idx;
    int current_leader;
    // Initialize indices
    std::vector<int> indices_left(number_of_points);
    std::vector<int> nearest_leader(number_of_points);
    std::vector<double> nearest_leader_distance(number_of_points);
    std::fill(nearest_leader_distance.begin(), nearest_leader_distance.end(), INFINITY);

    int number_of_leaders = 0;
    cluster_leaders.resize(0);
    cluster_leaders.reserve(number_of_clusters);

    std::iota(indices_left.begin(), indices_left.end(), 0);

    // Run an initial iteration without updates to get a good starting point from a random point.
    current_leader_indices_idx = randomInt(number_of_points);
    current_leader = indices_left[current_leader_indices_idx];
    int next_leader = current_leader;
    int next_leader_idx = 0;
    double next_distance = -INFINITY;

    for (size_t i_indices_left = 0; i_indices_left < indices_left.size(); ++i_indices_left)
    {
        int i = indices_left[i_indices_left];
        double distance_i_to_current_leader = distance(current_leader, i);
    
        // Is the new distance the farthest away so far?
        if (distance_i_to_current_leader < next_distance)
            continue;
        
        next_leader = i;
        next_leader_idx = i_indices_left;
        next_distance = distance_i_to_current_leader;
    }
    // Switch to the next leader.
    current_leader = next_leader;
    current_leader_indices_idx = next_leader_idx;
    // Remove next leader from indices left.
    size_t last_indices_left = indices_left.size() - 1;
    std::swap(indices_left[current_leader_indices_idx], indices_left[last_indices_left]);
    indices_left.resize(last_indices_left);
    // Add to cluster leaders
    cluster_leaders.push_back(current_leader);
    number_of_leaders += 1;
    // Set nearest for current leader
    nearest_leader[current_leader] = number_of_leaders - 1; 
    
    while (number_of_leaders < number_of_clusters)
    {
        next_leader = current_leader;
        next_leader_idx = 0;
        next_distance = -INFINITY;

        for (size_t i_indices_left = 0; i_indices_left < indices_left.size(); ++i_indices_left)
        {
            int i = indices_left[i_indices_left];
            double distance_i_to_current_leader = distance(current_leader, i);

            // Is the new distance better (and do we update the distance-to-nearest-list)
            if (distance_i_to_current_leader > nearest_leader_distance[i])
                continue;
            
            nearest_leader_distance[i] = distance_i_to_current_leader;
            // nearest_leader[i] = current_leader; // Identify cluster by leader index.
            nearest_leader[i] = number_of_leaders - 1; // Identify cluster by index in the order of which the leaders were assigned.
        
            // Is the new distance the farthest away so far?
            if (distance_i_to_current_leader < next_distance)
                continue;
            
            next_leader = i;
            next_leader_idx = i_indices_left;
            next_distance = distance_i_to_current_leader;
        }
        // Switch to the next leader.
        current_leader = next_leader;
        current_leader_indices_idx = next_leader_idx;
        // Remove next leader from indices left.
        size_t last_indices_left = indices_left.size() - 1;
        std::swap(indices_left[current_leader_indices_idx], indices_left[last_indices_left]);
        indices_left.resize(last_indices_left);
        // Add to cluster leaders
        cluster_leaders.push_back(current_leader);
        number_of_leaders += 1;
        // Set cluster of new leader
        nearest_leader[current_leader] = number_of_leaders - 1; 
    }
}

void mallocCurrentClusterSpace()
{
    population_cluster_sizes = (int*) Malloc(number_of_mixing_components * sizeof(int));
    population_indices_of_cluster_members = (int**) Malloc(sizeof(int*) * number_of_mixing_components);

    
    // Maybe use realloc for this instead.
    which_extreme                 = (int*) realloc(which_extreme, number_of_mixing_components*sizeof(int));

    objective_means_scaled        = (double **) realloc(objective_means_scaled, number_of_mixing_components*sizeof( double * ) );    
    for(int i = 0; i < number_of_mixing_components; i++ )
        objective_means_scaled[i] = (double *) realloc(objective_means_scaled[i], number_of_objectives*sizeof( double ) );
}

template <typename T, typename K>
void createKNNLeadersClustering(int number_of_clusters, int* leaders, K&& k_, T&& distance, int start_cluster=0, bool filter_zeros=false)
{
    std::vector<int> indices(population_size);
    std::vector<double> distances(population_size);
    for (int c_idx = start_cluster; c_idx < start_cluster + number_of_clusters; ++c_idx)
    {
        // Compute distances
        int leader = leaders[c_idx];
        int k = std::min(k_(leader), population_size);
        int filter_max_zeroes = k / 2; // good values are probably 1 or k / 2

        setVectorToRandomOrdering(indices);
        // std::iota(indices.begin(), indices.end(), 0);
        int num_zero_distances = 0;
        for (int j = 0; j < population_size; ++j)
        {
            distances[j] = distance(leader, j);
            num_zero_distances += distances[j] <= 0;
        }

        if (filter_zeros && num_zero_distances > filter_max_zeroes)
        {
            auto new_end = std::remove_if(indices.begin(), indices.end(),
            [&distances, &num_zero_distances, &filter_max_zeroes](int i)
            {
                if (distances[i] <= 0.0 && num_zero_distances > filter_max_zeroes)
                {
                    num_zero_distances -= 1;
                    return true;
                }
                return false;
            });
            indices.erase(new_end, indices.end());
        }

        // Compute
        std::nth_element(indices.begin(), indices.begin() + k, indices.end(), 
        [&distances](int a, int b)
            {
                return distances[a] < distances[b];
            });
        // Assert order is ascending.
        // assert(distances[indices[k - 1]] <= distances[indices[k]]);
        // assert(distances[indices[k]] <= distances[indices[k + 1]]);
        // std::cout << "Nearest for " << c_idx << ":" << distances[indices[0]] << "; farthest: " << distances[*(indices.end()-1)] << std::endl;
        int s = k;
        population_indices_of_cluster_members[c_idx] = (int*) Malloc(sizeof(int) * s);
        std::copy(indices.begin(), indices.begin() + k, population_indices_of_cluster_members[c_idx]);
        population_cluster_sizes[c_idx] = s;
        // Compute objective means.
        for (int o = 0; o < number_of_objectives; ++o)
        {
            objective_means_scaled[c_idx][o] = 0.0;
        }
        for (auto member_it = indices.begin(); member_it != indices.begin() + k; ++member_it)
        {
            int member = *member_it;
            for (int o = 0; o < number_of_objectives; ++o)
            {
                objective_means_scaled[c_idx][o] += objective_values[member][o];
            }
        }
        for (int o = 0; o < number_of_objectives; ++o)
        {
            // Objective means
            objective_means_scaled[c_idx][o] /= s;
            // Scaled.
            objective_means_scaled[c_idx][o] /= objective_ranges[o];
        }

    }
}

template <typename T, typename K>
void createSymmetricKNNClustering(int number_of_points, K&& k_, T&& distance, bool filter_zeros=false)
{
    std::vector<int> indices(population_size);
    std::vector<double> distances(population_size);
    std::vector<std::vector<int>> clusters(population_size); 

    for (int c_idx = 0; c_idx < number_of_points; ++c_idx)
    {
        // Compute distances
        int leader = c_idx;
        int k = std::min(k_(leader), population_size);
        int filter_max_zeroes = k / 2; // good values are probably 1 or k / 2
        setVectorToRandomOrdering(indices);
        // std::iota(indices.begin(), indices.end(), 0);

        int num_zero_distances = 0;
        for (int j = 0; j < population_size; ++j)
        {
            distances[j] = distance(leader, j);
            num_zero_distances += distances[j] <= 0;
        }

        if (filter_zeros && num_zero_distances > filter_max_zeroes)
        {
            auto new_end = std::remove_if(indices.begin(), indices.end(),
            [&distances, &num_zero_distances, &filter_max_zeroes](int i)
            {
                if (distances[i] <= 0.0 && num_zero_distances > filter_max_zeroes)
                {
                    num_zero_distances -= 1;
                    return true;
                }
                return false;
            });
            indices.erase(new_end, indices.end());
        }
        
        // Compute KNN.
        if (indices.size() >= static_cast<size_t>(k))
            std::nth_element(indices.begin(), indices.begin() + k, indices.end(), 
            [&distances](int a, int b)
                {
                    return distances[a] < distances[b];
                });
        // Assert order is ascending.
        // assert(distances[indices[k - 1]] <= distances[indices[k]]);
        // assert(distances[indices[k]] <= distances[indices[k + 1]]);

        clusters[leader].insert(clusters[leader].end(), indices.begin(), indices.begin() + k);
        // Additionally we include the leader in all clusters for which it is a KNN.
        for (auto it = indices.begin(); it != indices.begin() + k; ++it)
        {
            clusters[*it].push_back(leader);
        }
    }
    for (int c_idx = 0; c_idx < number_of_points; ++c_idx)
    {
        int leader = c_idx;
        auto& current_cluster = clusters[leader];

        // Remove duplicate entries.
        std::sort(current_cluster.begin(), current_cluster.end());
        auto new_end = std::unique(current_cluster.begin(), current_cluster.end());
        current_cluster.erase(new_end, current_cluster.end());

        int s = current_cluster.size();
        population_indices_of_cluster_members[c_idx] = (int*) Malloc(sizeof(int) * s);
        std::copy(current_cluster.begin(), current_cluster.begin() + s, population_indices_of_cluster_members[c_idx]);
        population_cluster_sizes[c_idx] = s;
        // Compute objective means.
        for (int o = 0; o < number_of_objectives; ++o)
        {
            objective_means_scaled[c_idx][o] = 0.0;
        }
        for (auto member_it = current_cluster.begin(); member_it != clusters[leader].begin() + s; ++member_it)
        {
            int member = *member_it;
            for (int o = 0; o < number_of_objectives; ++o)
            {
                objective_means_scaled[c_idx][o] += objective_values[member][o];
            }
        }
        for (int o = 0; o < number_of_objectives; ++o)
        {
            // Objective means
            objective_means_scaled[c_idx][o] /= s;
            // Scaled.
            objective_means_scaled[c_idx][o] /= objective_ranges[o];
        }

    }
}

void turnClusterAssignmentsIntoClusters(int number_of_elements, int number_of_cluster_assignments, int** cluster_assignments, int number_of_clusters)
{
    int total_number_of_clusters = 0;
    for (int cai = 0; cai < number_of_cluster_assignments; ++cai)
        total_number_of_clusters += number_of_clusters;

    // Ensure this is true by making it true.
    orig_number_of_mixing_components = number_of_mixing_components;
    number_of_mixing_components = total_number_of_clusters;

    population_cluster_sizes = (int*) Malloc(number_of_mixing_components * sizeof(int));
    population_indices_of_cluster_members = (int**) Malloc(sizeof(int*) * number_of_mixing_components);

    std::vector<std::vector<int>> cluster_memberships(number_of_clusters);
    for (int cai = 0; cai < number_of_cluster_assignments; ++cai)
    {
        // Collect clusters for this set of cluster assignments.
        
        for (int i = 0; i < number_of_elements; ++i)
        {
            int cluster_index = cluster_assignments[cai][i];
            if (cluster_index >= 0)
                cluster_memberships[cluster_index].push_back(i);
        }
    }

    for (int cc_idx = 0; cc_idx < number_of_clusters; ++cc_idx)
    {
        int c_idx = cc_idx;
        int s = cluster_memberships[cc_idx].size();
        population_indices_of_cluster_members[c_idx] = (int*) Malloc(sizeof(int) * s);
        std::copy(cluster_memberships[cc_idx].begin(), cluster_memberships[cc_idx].end(), population_indices_of_cluster_members[c_idx]);
        population_cluster_sizes[c_idx] = s;
        // Compute objective means.
        for (int o = 0; o < number_of_objectives; ++o)
        {
            objective_means_scaled[c_idx][o] = 0.0;
        }
        for (int member: cluster_memberships[cc_idx])
        {
            for (int o = 0; o < number_of_objectives; ++o)
            {
                objective_means_scaled[c_idx][o] += objective_values[member][o];
            }
        }
        for (int o = 0; o < number_of_objectives; ++o)
        {
            // Objective means
            objective_means_scaled[c_idx][o] /= s;
            // Scaled.
            objective_means_scaled[c_idx][o] /= objective_ranges[o];
        }
    }
 
}

void performGSSHammingClustering()
{
    std::vector<int> indices_left(population_size);
    setVectorToRandomOrdering(indices_left);
    // std::iota(indices_left.begin(), indices_left.end(), 0);
    std::vector<int> cluster_assignment(population_size);

    performGSSClustering(population_size, number_of_mixing_components, 
    [](int leader, int i)
    {
        int hamming = hammingDistanceInParameterSpace(population[leader], population[i]);
        double normalized_hamming = static_cast<double>(hamming) / static_cast<double>(number_of_parameters);
        return normalized_hamming;
    }, indices_left, cluster_assignment);

    int* single_cluster_assignment = &cluster_assignment[0];
    turnClusterAssignmentsIntoClusters(population_size, 1, &single_cluster_assignment, number_of_mixing_components);
}

void performKNNGSSHammingClustering()
{
    int k = std::min(population_size, (population_size * 2) / number_of_mixing_components);
    // Define distance
    auto distance = [](int leader, int i)
    {
        int hamming = hammingDistanceInParameterSpace(population[leader], population[i]);
        double normalized_hamming = static_cast<double>(hamming) / static_cast<double>(number_of_parameters);
        return normalized_hamming;
    };

    // Select leaders
    std::vector<int> cluster_leaders;
    performGreedySubsetScatteringWithDistance(population_size, number_of_mixing_components, distance, cluster_leaders);
    // Apply KNN clustering
    mallocCurrentClusterSpace();
    createKNNLeadersClustering(cluster_leaders.size(), cluster_leaders.data(), [k](int){ return k; }, distance);
}

void performHierarchicalKNNGSSHammingClustering()
{
    // Some notes about this.
    // The first 3 leaders get a cluster with k = population_size * 2 / 3
    // Then the first 6 (again, the first 3 leaders as well) with k = population_size * 2 / 6
    // Then the first 12 (again, the first 6 leaders as well) with k = population_size * 2 / 12
    // And so on.
    // This approach uses the initial value of number_of_mixing_components as a guidance, but it may overshoot
    // and adjusts the value accordingly.

    int current_amount = 3;
    int number_of_mixing_components_so_far = current_amount;
    int target_number_of_mixing_components = std::min(population_size / 4, 2 * number_of_mixing_components);
    while (number_of_mixing_components_so_far < target_number_of_mixing_components)
    {
        current_amount *= 2;
        number_of_mixing_components_so_far += current_amount;
    }

    orig_number_of_mixing_components = number_of_mixing_components;
    number_of_mixing_components = number_of_mixing_components_so_far;

    // Define distance
    auto distance = [](int leader, int i)
    {
        int hamming = hammingDistanceInParameterSpace(population[leader], population[i]);
        double normalized_hamming = static_cast<double>(hamming) / static_cast<double>(number_of_parameters);
        return normalized_hamming;
    };

    // Select leaders
    std::vector<int> cluster_leaders;
    performGreedySubsetScatteringWithDistance(population_size, number_of_mixing_components, distance, cluster_leaders);
    // Apply Hierarchical KNN clustering
    initializeClusters();
    mallocCurrentClusterSpace();
    int acc = 0;
    int t = 3;
    while (acc < number_of_mixing_components)
    {
        int k = std::min(population_size, (population_size * 2) / t);
        createKNNLeadersClustering(t, cluster_leaders.data(), [k](int){ return k; }, distance, acc);
        acc += t;
        t *= 2;
    }
}

void performGSSHammingInvHammingClustering()
{
    std::vector<int> indices_left(population_size);
    setVectorToRandomOrdering(indices_left);
    // std::iota(indices_left.begin(), indices_left.end(), 0);
    std::vector<int> cluster_assignment(population_size);

    performGSSClustering(population_size, number_of_mixing_components, 
    [](int leader, int i)
    {
        int hamming = hammingDistanceInParameterSpace(population[leader], population[i]);
        double normalized_hamming = static_cast<double>(hamming) / static_cast<double>(number_of_parameters);
        return normalized_hamming * (1 - normalized_hamming);
    }, indices_left, cluster_assignment);

    int* single_cluster_assignment = &cluster_assignment[0];
    turnClusterAssignmentsIntoClusters(population_size, 1, &single_cluster_assignment, number_of_mixing_components);
}

int sampleRandomClusteringKNNk1()
{
    double min_el = 8;

    double fmax = population_size / min_el;
    return std::ceil(population_size / std::ceil(randomRealUniform01() * fmax + 1e-6));
}

int sampleRandomClusteringKNNk2(int solution_idx)
{
    if (solution_k[solution_idx] == 0 || s_NIS[solution_idx] > 0)
    {
        int orig_k = solution_k[solution_idx];
        solution_k[solution_idx] = sampleRandomClusteringKNNk1();
        std::cout << "(Re)drawing " << solution_idx << ": " << orig_k << " --> " << solution_k[solution_idx] << "/" << population_size << '\n';
    }
    return solution_k[solution_idx];
}

int sampleRandomClusteringKNNk3(int solution_idx)
{
    if (solution_k[solution_idx] == 0)
    {
        solution_k[solution_idx] = sampleRandomClusteringKNNk1();
        // std::cout << "Drawing initial " << solution_idx << ": " << solution_k[solution_idx] << "/" << population_size << '\n';
    }
    else if (s_NIS[solution_idx] > 0)
    {
        int orig_k = solution_k[solution_idx];
        int tries = 0;
        while (solution_k[solution_idx] == orig_k && tries < 4)
        {
            if (randomRealUniform01() < 0.25)
            {
                solution_k[solution_idx] = sampleRandomClusteringKNNk1();
            }
            else
            {
                solution_k[solution_idx] = solution_k[randomInt(population_size)];
            }
            tries += 1;
        }
        std::cout << "Redrawing " << solution_idx << ": " << orig_k << " --> " << solution_k[solution_idx] << "/" << population_size << '\n';
    }
    return solution_k[solution_idx];
}

int getClusteringKNNk(int mode, int solution_idx)
{
    if (mode == -1) return std::ceil(std::log2(population_size));
    if (mode == -2) return sampleRandomClusteringKNNk1();
    if (mode == -3) return sampleRandomClusteringKNNk2(solution_idx);
    if (mode == -4) return sampleRandomClusteringKNNk3(solution_idx);

    // Default
    return std::ceil(std::sqrt(population_size));
}

void performAllKNNHammingClustering(bool filter=false)
{
    auto k = [](int solution_idx){ return getClusteringKNNk(clustering_mode_modifier, solution_idx); };

    // Define distance
    auto distance = [](int leader, int i)
    {
        int hamming = hammingDistanceInParameterSpace(population[leader], population[i]);
        double normalized_hamming = static_cast<double>(hamming) / static_cast<double>(number_of_parameters);
        return normalized_hamming;
    };
    orig_number_of_mixing_components = number_of_mixing_components;
    number_of_mixing_components = population_size;
    initializeClusters();
    mallocCurrentClusterSpace();
    mixture_assignment.resize(population_size);
    std::iota(mixture_assignment.begin(), mixture_assignment.end(), 0);
    createKNNLeadersClustering(population_size, mixture_assignment.data(), k, distance, 0, filter);
}

void performAllKNNHammingSymmetricClustering(bool filter=false)
{
    auto k = [](int solution_idx){ return getClusteringKNNk(clustering_mode_modifier, solution_idx); };

    // Define distance
    auto distance = [](int leader, int i)
    {
        int hamming = hammingDistanceInParameterSpace(population[leader], population[i]);
        double normalized_hamming = static_cast<double>(hamming) / static_cast<double>(number_of_parameters);
        return normalized_hamming;
    };
    orig_number_of_mixing_components = number_of_mixing_components;
    number_of_mixing_components = population_size;
    initializeClusters();
    mallocCurrentClusterSpace();
    createSymmetricKNNClustering(population_size, k, distance, filter);
    mixture_assignment.resize(population_size);
    std::iota(mixture_assignment.begin(), mixture_assignment.end(), 0);
}

void performClustering(int mode)
{
    // std::cout << "Performing clustering with mode " << mode << std::endl;

    if (preserve_extreme_frontal_points)
    {
        determineRanks();
        insertExtremeArchiveSolution();
    }

    mixture_assignment.resize(0);

    switch (mode)
    {
    case CLUSTER_MODE_DEFAULT:
        initializeClusters();
        performClusteringDefault();
        break;
    case CLUSTER_MODE_LINE:
        initializeClusters();
        determineRanks();
        determineLines();
        computeNearestLineForEachIndividual();
        assignSolutionsToLines();
        performClusteringLine();
        break;
    case CLUSTER_MODE_LINE_KNN:
        initializeClusters();
        determineRanks();
        determineLines();
        computeNearestLineForEachIndividual();
        assignSolutionsToLines();
        performClusteringLineKNN();
        break;
    case CLUSTER_MODE_COMPUTE_LINES_ONLY:
        initializeClusters();
        determineRanks();
        determineLines();
        computeNearestLineForEachIndividual();
        assignSolutionsToLines();
        break;
    case CLUSTER_MODE_HAMMING_GSS:
        initializeClusters();
        performGSSHammingClustering();
        break;
    case CLUSTER_MODE_HAMMING_INV_HAMMING_GSS:
        initializeClusters();
        performGSSHammingInvHammingClustering();
        break;
    case CLUSTER_MODE_HAMMING_KNN_GSS:
        initializeClusters();
        performKNNGSSHammingClustering();
        break;
    case CLUSTER_MODE_HAMMING_HIERARCHICAL_KNN_GSS:
        performHierarchicalKNNGSSHammingClustering();
        break;
    case CLUSTER_MODE_HAMMING_KNN_ALL:
        performAllKNNHammingClustering();
        break;
    case CLUSTER_MODE_HAMMING_KNN_SYM_ALL:
        performAllKNNHammingSymmetricClustering();
        break;
    case CLUSTER_MODE_HAMMING_FILTERKNN_ALL:
        performAllKNNHammingClustering(true);
        break;
    case CLUSTER_MODE_HAMMING_FILTERKNN_SYM_ALL:
        performAllKNNHammingSymmetricClustering(true);
        break;
    default:
        break;
    }

    // find extreme-region clusters
    determineExtremeClusters(number_of_mixing_components, objective_means_scaled, which_extreme);
    determineExtremeKernels();
}

void learnLinkageOnCurrentPopulation()
{
    // learn linkage tree for every cluster
    for(int i = 0; i < number_of_mixing_components; i++ )
        learnLinkageTree( i );

}

double weightedTchebycheff(size_t n, double* a, double* b, double* w, double* vr)
{
    double r = 0;
    for (size_t i = 0; i < n; ++i)
    {
        if (vr == NULL)
        {
            r = std::max(r, std::abs(a[i] - b[i]) / (vr[i] * w[i]));
        }
        else
        {
            r = std::max(r, std::abs(a[i] - b[i]) / w[i]);
        }
    }
    return r;
}

void computeObjectiveMinMaxRangesNadirAndUtopianPoint()
{
    objective_min.resize(number_of_objectives);
    objective_max.resize(number_of_objectives);
    
    archive_objective_min.resize(number_of_objectives);
    archive_objective_max.resize(number_of_objectives);
    
    utopian_point.resize(number_of_objectives);
    nadir_point.resize(number_of_objectives);

    adjusted_nadir.resize(number_of_objectives);

    objective_range.resize(number_of_objectives);
    std::fill(objective_min.begin(), objective_min.end(), INFINITY);
    std::fill(objective_max.begin(), objective_max.end(), -INFINITY);
    std::fill(archive_objective_min.begin(), archive_objective_min.end(), INFINITY);
    std::fill(archive_objective_max.begin(), archive_objective_max.end(), -INFINITY);

    // Population
    for (size_t i = 0; i < (size_t) population_size; ++i)
    {
        for (size_t o = 0; o < (size_t) number_of_objectives; ++o)
        {
            objective_min[o] = std::min(objective_min[o], objective_values[i][o]);
            objective_max[o] = std::max(objective_max[o], objective_values[i][o]);
        }
    }
    // Archive
    for (size_t i = 0; i < (size_t) elitist_archive_size; ++i)
    {
        for (size_t o = 0; o < (size_t) number_of_objectives; ++o)
        {
            archive_objective_min[o] = std::min(archive_objective_min[o], elitist_archive_objective_values[i][o]);
            archive_objective_max[o] = std::max(archive_objective_max[o], elitist_archive_objective_values[i][o]);
        }
    }

    for (size_t o = 0; o < (size_t) number_of_objectives; ++o)
    {
        // std::cout << "Bounds for objective " << o << ". min: " << objective_min[o] << ". max: " << objective_max[o] << ".\n";
        objective_range[o] = objective_max[o] - objective_min[o];

        if (use_nadir_of_population)
        {
            utopian_point[o] = optimization[o] == MINIMIZATION ? objective_min[o] : objective_max[o];
            nadir_point[o] = optimization[o] == MINIMIZATION ? objective_max[o] : objective_min[o];
        }
        else
        {
            utopian_point[o] = optimization[o] == MINIMIZATION ? archive_objective_min[o] : archive_objective_max[o];
            nadir_point[o] = optimization[o] == MINIMIZATION ? archive_objective_max[o] : archive_objective_min[o];
        }

        adjusted_nadir[o] = nadir_point[o] + (optimization[o] == MAXIMIZATION ? -1 : 1) * objective_range[o] * adjusted_nadir_adjustment;
    }
}

int nondominated_sort(int n_points, double** points, int* rank, 
    std::optional<std::reference_wrapper<std::vector<int>>> order,
    std::optional<std::reference_wrapper<std::vector<std::pair<int, int>>>> order_start_stop)
{
    std::vector<std::vector<size_t>> dominates(n_points);
    std::vector<size_t> domination_counts(n_points);
    int current_rank_idx = 0;
    std::vector<size_t> current_rank;
    std::vector<size_t> next_rank;
    
    // Determine domination counts
    for (size_t i = 0; i < (size_t) n_points; ++i)
    {
        size_t domination_count = 0;
        for (size_t j = 0; j < (size_t) n_points; ++j)
        {
            // TODO: Constraints!
            if (paretoDominates(points[j], points[i]))
            {
                domination_count += 1;
                dominates[j].push_back(i);
            }
        }
        domination_counts[i] = domination_count;
        if (domination_count == 0)
        {
            // Undominated! Part of next front.
            current_rank.push_back(i);
        }
    }

    while(current_rank.size() > 0)
    {
        int order_idx_start = 0;
        int order_idx_end = 0;
        // Append current list to ordering list.
        if (order.has_value())
        {
            std::vector<int> &order_v = order.value();
            order_idx_start = order_v.size();
            order_v.insert(order_v.end(), current_rank.begin(), current_rank.end());
            order_idx_end = order_v.size();
        }
        if (order.has_value() && order_start_stop.has_value())
        {
            std::vector<std::pair<int, int>> &order_start_stop_v = order_start_stop.value();
            order_start_stop_v.push_back(std::make_pair(order_idx_start, order_idx_end));
        }

        for (size_t i: current_rank)
        {
            rank[i] = current_rank_idx;
            for (size_t j: dominates[i])
            {
                domination_counts[j] -= 1;
                if (domination_counts[j] == 0)
                {
                    next_rank.push_back(j);
                }
            }
        }
        current_rank_idx += 1;
        current_rank.clear();
        current_rank.swap(next_rank);
    }

    return current_rank_idx;
}

int** clustering(double **objective_values_pool, int pool_size, int number_of_dimensions, 
                    int number_of_clusters, int *pool_cluster_size, double** objective_means_scaled )
{
    int i, j, k, j_min, number_to_select,
        *pool_indices_of_leaders, *k_means_cluster_sizes, **pool_indices_of_cluster_members_k_means,
        **pool_indices_of_cluster_members, size_of_one_cluster;
    double distance, distance_smallest, epsilon,
            **objective_values_pool_scaled, **objective_means_scaled_new, *distances_to_cluster;
            
    if (number_of_clusters > 1)
        *pool_cluster_size   = (2*pool_size)/number_of_clusters;
    else
    {
        *pool_cluster_size   = pool_size;
        pool_indices_of_cluster_members = (int**)Malloc(number_of_clusters * sizeof(int*));
        pool_indices_of_cluster_members[0] = (int*)Malloc(pool_size * sizeof(int));
        for(i = 0; i < pool_size; i++)
            pool_indices_of_cluster_members[0][i] = i;
        return (pool_indices_of_cluster_members);
    }

    size_of_one_cluster  = *pool_cluster_size;

    /* Determine the leaders */
    objective_values_pool_scaled = (double **) Malloc( pool_size*sizeof( double * ) );
    for( i = 0; i < pool_size; i++ )
        objective_values_pool_scaled[i] = (double *) Malloc( number_of_dimensions*sizeof( double ) );
    for( i = 0; i < pool_size; i++ )
        for( j = 0; j < number_of_dimensions; j++ )
            objective_values_pool_scaled[i][j] = objective_values_pool[i][j]/objective_ranges[j];

    /* Heuristically find k far-apart leaders */
    number_to_select             = number_of_clusters;
    pool_indices_of_leaders = greedyScatteredSubsetSelection( objective_values_pool_scaled, pool_size, number_of_dimensions, number_to_select );

    for( i = 0; i < number_of_clusters; i++ )
        for( j = 0; j < number_of_dimensions; j++ )
            objective_means_scaled[i][j] = objective_values_pool[pool_indices_of_leaders[i]][j]/objective_ranges[j];

    /* Perform k-means clustering with leaders as initial mean guesses */
    objective_means_scaled_new = (double **) Malloc( number_of_clusters*sizeof( double * ) );
    for( i = 0; i < number_of_clusters; i++ )
        objective_means_scaled_new[i] = (double *) Malloc( number_of_dimensions*sizeof( double ) );

    pool_indices_of_cluster_members_k_means = (int **) Malloc( number_of_clusters*sizeof( int * ) );
    for( i = 0; i < number_of_clusters; i++ )
        pool_indices_of_cluster_members_k_means[i] = (int *) Malloc( pool_size*sizeof( int ) );

    k_means_cluster_sizes = (int *) Malloc( number_of_clusters*sizeof( int ) );

    epsilon = 1e+308;
    while( epsilon > 1e-10 )
    {
        for( j = 0; j < number_of_clusters; j++ )
        {
            k_means_cluster_sizes[j] = 0;
            for( k = 0; k < number_of_dimensions; k++ )
                objective_means_scaled_new[j][k] = 0.0;
        }

        for( i = 0; i < pool_size; i++ )
        {
            j_min             = -1;
            distance_smallest = -1;
            for( j = 0; j < number_of_clusters; j++ )
            {
                distance = distanceEuclidean( objective_values_pool_scaled[i], objective_means_scaled[j], number_of_dimensions );
                if( (distance_smallest < 0) || (distance < distance_smallest) )
                {
                    j_min             = j;
                    distance_smallest = distance;
                }
            }
            pool_indices_of_cluster_members_k_means[j_min][k_means_cluster_sizes[j_min]] = i;
            for( k = 0; k < number_of_dimensions; k++ )
                objective_means_scaled_new[j_min][k] += objective_values_pool_scaled[i][k];
            k_means_cluster_sizes[j_min]++;
        }

        for( j = 0; j < number_of_clusters; j++ )
            for( k = 0; k < number_of_dimensions; k++ )
                objective_means_scaled_new[j][k] /= (double) k_means_cluster_sizes[j];

        epsilon = 0;
        for( j = 0; j < number_of_clusters; j++ )
        {
            epsilon += distanceEuclidean( objective_means_scaled[j], objective_means_scaled_new[j], number_of_dimensions );
            for( k = 0; k < number_of_dimensions; k++ )
                objective_means_scaled[j][k] = objective_means_scaled_new[j][k];
        }
    }

    /* Shrink or grow the result of k-means clustering to get the final equal-sized clusters */
    pool_indices_of_cluster_members = (int**)Malloc(number_of_clusters * sizeof(int*));
    distances_to_cluster = (double *) Malloc( pool_size*sizeof( double ) );
    for( i = 0; i < number_of_clusters; i++ )
    {
        for( j = 0; j < pool_size; j++ )
            distances_to_cluster[j] = distanceEuclidean( objective_values_pool_scaled[j], objective_means_scaled[i], number_of_dimensions );

        for( j = 0; j < k_means_cluster_sizes[i]; j++ )
            distances_to_cluster[pool_indices_of_cluster_members_k_means[i][j]] = 0;

        pool_indices_of_cluster_members[i]          = mergeSort( distances_to_cluster, pool_size );
    }

    // Re-calculate clusters' means
    for( i = 0; i < number_of_clusters; i++)
    {
        for (j = 0; j < number_of_dimensions; j++)
            objective_means_scaled[i][j] = 0.0;

        for (j = 0; j < size_of_one_cluster; j++)
        {
            for( k = 0; k < number_of_dimensions; k++)
                objective_means_scaled[i][k] +=
                    objective_values_pool_scaled[pool_indices_of_cluster_members[i][j]][k];
        }

        for (j = 0; j < number_of_dimensions; j++)
        {
            objective_means_scaled[i][j] /= (double) size_of_one_cluster;
        }
    }

    free( distances_to_cluster );
    free( k_means_cluster_sizes );
    for( i = 0; i < number_of_clusters; i++ )
        free( pool_indices_of_cluster_members_k_means[i] );
    free( pool_indices_of_cluster_members_k_means );
    for( i = 0; i < number_of_clusters; i++ )
        free( objective_means_scaled_new[i] );
    free( objective_means_scaled_new );
    for( i = 0; i < pool_size; i++ )
        free( objective_values_pool_scaled[i] );
    free( objective_values_pool_scaled );
    free( pool_indices_of_leaders );   

    return (pool_indices_of_cluster_members);
}

/**
 * greedyScatteredSubsetSelection with a subset limitation. 
 **/
int *greedyScatteredSubSubsetSelection(std::vector<int> &indices, double **points, int number_of_dimensions, int number_to_select, double* ranges = NULL )
{
    int     i, index_of_farthest, random_dimension_index, number_selected_so_far,
            *indices_left, *result;
    double *nn_distances, distance_of_farthest, value;

    if( number_to_select > (int) indices.size() )
    {
        printf("\n");
        printf("Error: greedyScatteredSubsetSelection asked to select %d solutions from set of size %ld.", number_to_select, indices.size());
        printf("\n\n");

        exit( 0 );
    }

    result = (int *) Malloc( number_to_select*sizeof( int ) );

    indices_left = (int *) Malloc( indices.size()*sizeof( int ) );
    for( i = 0; i < (int) indices.size(); i++ )
        indices_left[i] = indices[i];

    /* Find the first point: maximum value in a randomly chosen dimension */
    random_dimension_index = randomInt( number_of_dimensions );

    index_of_farthest    = 0;
    distance_of_farthest = points[indices_left[index_of_farthest]][random_dimension_index];
    for( i = 1; i < (int) indices.size() ; i++ )
    {
        if( points[indices_left[i]][random_dimension_index] > distance_of_farthest )
        {
            index_of_farthest    = i;
            distance_of_farthest = points[indices_left[i]][random_dimension_index];
        }
    }

    number_selected_so_far          = 0;
    result[number_selected_so_far]  = indices_left[index_of_farthest];
    indices_left[index_of_farthest] = indices_left[indices.size() -number_selected_so_far-1];
    number_selected_so_far++;

    /* Then select the rest of the solutions: maximum minimum
     * (i.e. nearest-neighbour) distance to so-far selected points */
    nn_distances = (double *) Malloc( indices.size() *sizeof( double ) );
    for( i = 0; i < (int) indices.size() -number_selected_so_far; i++ )
        nn_distances[i] = distanceEuclidean( points[indices_left[i]], points[result[number_selected_so_far-1]], number_of_dimensions, ranges );

    while( number_selected_so_far < number_to_select )
    {
        index_of_farthest    = 0;
        distance_of_farthest = nn_distances[0];
        for( i = 1; i < (int) indices.size() -number_selected_so_far; i++ )
        {
            if( nn_distances[i] > distance_of_farthest )
            {
                index_of_farthest    = i;
                distance_of_farthest = nn_distances[i];
            }
        }

        result[number_selected_so_far]  = indices_left[index_of_farthest];
        indices_left[index_of_farthest] = indices_left[indices.size()-number_selected_so_far-1];
        nn_distances[index_of_farthest] = nn_distances[indices.size()-number_selected_so_far-1];
        number_selected_so_far++;

        for( i = 0; i < (int) indices.size()-number_selected_so_far; i++ )
        {
            value = distanceEuclidean( points[indices_left[i]], points[result[number_selected_so_far-1]], number_of_dimensions, ranges );
            if( value < nn_distances[i] )
                nn_distances[i] = value;
        }
    }

    free( nn_distances );
    free( indices_left );
    return( result );
}

/**
 * Selects n points from a set of points. A
 * greedy heuristic is used to find a good
 * scattering of the selected points. First,
 * a point is selected with a maximum value
 * in a randomly selected dimension. The
 * remaining points are selected iteratively.
 * In each iteration, the point selected is
 * the one that maximizes the minimal distance
 * to the points selected so far.
 */
int *greedyScatteredSubsetSelection( double **points, int number_of_points, int number_of_dimensions, int number_to_select )
{
    int     i, index_of_farthest, random_dimension_index, number_selected_so_far,
            *indices_left, *result;
    double *nn_distances, distance_of_farthest, value;

    if( number_to_select > number_of_points )
    {
        printf("\n");
        printf("Error: greedyScatteredSubsetSelection asked to select %d solutions from set of size %d.", number_to_select, number_of_points);
        printf("\n\n");

        exit( 0 );
    }

    result = (int *) Malloc( number_to_select*sizeof( int ) );

    indices_left = (int *) Malloc( number_of_points*sizeof( int ) );
    for( i = 0; i < number_of_points; i++ )
        indices_left[i] = i;

    /* Find the first point: maximum value in a randomly chosen dimension */
    random_dimension_index = randomInt( number_of_dimensions );

    index_of_farthest    = 0;
    distance_of_farthest = points[indices_left[index_of_farthest]][random_dimension_index];
    for( i = 1; i < number_of_points; i++ )
    {
        if( points[indices_left[i]][random_dimension_index] > distance_of_farthest )
        {
            index_of_farthest    = i;
            distance_of_farthest = points[indices_left[i]][random_dimension_index];
        }
    }

    number_selected_so_far          = 0;
    result[number_selected_so_far]  = indices_left[index_of_farthest];
    indices_left[index_of_farthest] = indices_left[number_of_points-number_selected_so_far-1];
    number_selected_so_far++;

    /* Then select the rest of the solutions: maximum minimum
     * (i.e. nearest-neighbour) distance to so-far selected points */
    nn_distances = (double *) Malloc( number_of_points*sizeof( double ) );
    for( i = 0; i < number_of_points-number_selected_so_far; i++ )
        nn_distances[i] = distanceEuclidean( points[indices_left[i]], points[result[number_selected_so_far-1]], number_of_dimensions );

    while( number_selected_so_far < number_to_select )
    {
        index_of_farthest    = 0;
        distance_of_farthest = nn_distances[0];
        for( i = 1; i < number_of_points-number_selected_so_far; i++ )
        {
            if( nn_distances[i] > distance_of_farthest )
            {
                index_of_farthest    = i;
                distance_of_farthest = nn_distances[i];
            }
        }

        result[number_selected_so_far]  = indices_left[index_of_farthest];
        indices_left[index_of_farthest] = indices_left[number_of_points-number_selected_so_far-1];
        nn_distances[index_of_farthest] = nn_distances[number_of_points-number_selected_so_far-1];
        number_selected_so_far++;

        for( i = 0; i < number_of_points-number_selected_so_far; i++ )
        {
            value = distanceEuclidean( points[indices_left[i]], points[result[number_selected_so_far-1]], number_of_dimensions );
            if( value < nn_distances[i] )
                nn_distances[i] = value;
        }
    }

    free( nn_distances );
    free( indices_left );
    return( result );
}

void determineExtremeClusters(int number_of_mixing_components, double** objective_means_scaled, int* which_extreme)
{
    int i,j, index_best, *order;
    // find extreme clusters
    order = createRandomOrdering(number_of_objectives);
        
    for (i = 0; i < number_of_mixing_components; i++)
        which_extreme[i] = -1;  // not extreme cluster
    
    if(number_of_mixing_components > 1)
    {
        for (i = 0; i < number_of_objectives; i++)
        {
            index_best = -1;
        
            for (j = 0; j < number_of_mixing_components; j++)
            {
                if(optimization[order[i]] == MINIMIZATION)
                {
                    if( ((index_best == -1) || (objective_means_scaled[j][order[i]] < objective_means_scaled[index_best][order[i]]) )&&
                        (which_extreme[j] == -1) )
                        index_best = j;
                }
                else if(optimization[order[i]] == MAXIMIZATION)
                {
                    if( ((index_best == -1) || (objective_means_scaled[j][order[i]] > objective_means_scaled[index_best][order[i]]) )&&
                        (which_extreme[j] == -1) )
                        index_best = j;
                }
            }
            which_extreme[index_best] = order[i];
        }
    }

    free(order);
}

bool is_objective_a_better(double obj_a, double obj_b, int o)
{
    return (optimization[o] == MAXIMIZATION) ?
        (obj_a > obj_b) :
        (obj_a < obj_b);
}

void determineMixingPoolForSolution(int solution_index, int cluster_index, int /* line_index */, std::vector<int> &donor_indices);

void determineExtremeKernels()
{
    // find extreme clusters

    is_extreme_kernel.resize(population_size);
        
    for (int i = 0; i < population_size; i++)
        is_extreme_kernel[i] = -1;

    switch (extreme_kernel_mode)
    {
        // Simulate the old extreme clusters
        case -1:
        {
            // Get the original number of mixing components.
            int num_clust = orig_number_of_mixing_components;
            if (num_clust == -1)
                num_clust = number_of_mixing_components;

            // Prepare some memory for the scaled objectives.
            double** local_objective_means_scaled = (double**) Malloc(sizeof(double*) * num_clust);
            for (int c = 0; c < num_clust; ++c)
                local_objective_means_scaled[c] = (double*) Malloc(sizeof(double) * number_of_objectives);
            
            // Perform original clustering approach in objective space.
            int size_of_one_cluster;
            int** clusters = clustering(objective_values, population_size, number_of_objectives, 
                            num_clust, &size_of_one_cluster, local_objective_means_scaled);
        

            // Determine which of these clusters are extreme.
            int* which_extreme = (int*) Malloc(sizeof(int) * num_clust);
            determineExtremeClusters(num_clust, local_objective_means_scaled, which_extreme);

            // Cleanup local_objective_means_scaled
            for (int c = 0; c < num_clust; ++c)
                free(local_objective_means_scaled[c]);
            free(local_objective_means_scaled);

            // Simulate drawing a random sample from the set of clusters
            // each sample is contained in (without materializing the list)
            std::vector<int> clust_count(population_size);
            std::fill(clust_count.begin(), clust_count.end(), 0);
            for (int c = 0; c < num_clust; ++c)
            {
                int* cc = clusters[c];
                for (int idx = 0; idx < size_of_one_cluster; ++idx)
                {
                    // Increase the cluster count.
                    int i = cc[idx];
                    clust_count[i] += 1;
                    // If...
                    if (clust_count[i] == 1 || // This is the first time `i` occurs in a cluster (to avoid drawing a random number)
                        which_extreme[c] == is_extreme_kernel[i] || // The operation below is a no-op (to avoid drawing a random number)
                        randomInt(clust_count[i]) == 0) // Or a random number indicates replacement of current best.
                        is_extreme_kernel[i] = which_extreme[c]; // Set the extreme kernel to the objective this cluster is the extreme of.
                }

                // Free the memory allocated by clustering, we no longer need this subset.
                free(cc);
            }

            // We no longer need the array either.
            free(clusters);
            free(which_extreme);
        }
        break;
        // No extreme kernels
        case 0: break;
        // Has most extreme value 
        case 1:
        {
            for (int o = 0; o < number_of_objectives; ++o)
            {
                int best_idx = 0;
                double best_obj_value = (optimization[o] == MAXIMIZATION) ? -INFINITY : INFINITY;
                for (int i = 0; i < population_size; ++i)
                {
                    double current_obj_value = objective_values[i][o];
                    if ( is_objective_a_better(current_obj_value, best_obj_value, o) )
                    {
                        best_idx = i;
                        best_obj_value = current_obj_value;   
                    }
                }
                is_extreme_kernel[best_idx] = o;
            }
        }
        break;
        // Is within the extreme_radius solutions with the best value
        // If within extreme for multiple. Choose randomly!
        case 2: case 3: case 4:
        {
            int k;
            if (extreme_kernel_mode == 2)
                k = extreme_radius * population_size;
            else if (extreme_kernel_mode == 3)
                k = std::ceil(std::sqrt(population_size));
            else
                k = std::ceil(2 * std::sqrt(population_size));
            std::vector<int> num_extremes(population_size);
            std::fill(num_extremes.begin(), num_extremes.end(), 0);
            std::vector<int> indices(population_size);
            for (int o = 0; o < number_of_objectives; ++o)
            {
                setVectorToRandomOrdering(indices);
                std::nth_element(indices.begin(), indices.begin() + k, indices.end(),
                    [&o](int idx_a, int idx_b)
                    {
                        return is_objective_a_better(objective_values[idx_a][o], objective_values[idx_b][o], o);
                    });
                // Assert sortedness.
                // for (int i_before = 0; i_before < k; i_before++)
                //     for (int i_after = k + 1; i_after < population_size; i_after++)
                //         assert(!is_objective_a_better(objective_values[indices[i_after]][o], objective_values[indices[i_before]][o], 0));
                
                for (int indices_idx = 0; indices_idx < k; ++indices_idx)
                {
                    int i = indices[indices_idx];
                    num_extremes[i] += 1;
                    if (num_extremes[i] == 1 || randomInt(num_extremes[i]) == 0)
                    {
                        is_extreme_kernel[i] = o;
                    }
                }
            }

        }
        break;
        // By rank, mark best.
        case 5:
        {
            determineRanks();
            
            for (int o = 0; o < number_of_objectives; ++o)
            {
                int idx = 0;
                for (int r = 0; r < max_rank; ++r)
                {
                    int best_idx = 0;
                    double best_obj_value = (optimization[o] == MAXIMIZATION) ? -INFINITY : INFINITY;
                    for (; idx < population_size && rank[rank_order[idx]] == r; ++idx)
                    {
                        int i = rank_order[idx];
                        double current_obj_value = objective_values[i][o];
                        if ( is_objective_a_better(current_obj_value, best_obj_value, o) )
                        {
                            best_idx = i;
                            best_obj_value = current_obj_value;   
                        }
                    }
                    is_extreme_kernel[best_idx] = o;
                }
            }
        }
        break;
        // By rank, mark sqrt(2 * |R|) best
        case 6:
        {
            determineRanks();
            
            std::vector<int> num_extremes(population_size);
            std::fill(num_extremes.begin(), num_extremes.end(), 0);
            std::vector<int> indices;
            indices.reserve(population_size);
            int idx = 0;
            for (int r = 0; r < max_rank; ++r)
            {
                indices.clear();
                // Collect indices for current rank.
                for (; idx < population_size && rank[rank_order[idx]] == r; ++idx)
                {
                    int i = rank_order[idx];
                    indices.push_back(i);
                }
                int k = std::min(static_cast<double>(indices.size()), std::ceil(2 * std::sqrt(indices.size())));
                for (int o = 0; o < number_of_objectives; ++o)
                {
                    std::nth_element(indices.begin(), indices.begin() + k, indices.end(),
                        [&o](int idx_a, int idx_b)
                        {
                            return is_objective_a_better(objective_values[idx_a][o], objective_values[idx_b][o], o);
                        });

                    for (int idx = 0; idx < k; ++idx)
                    {
                        int i = indices[idx];
                        num_extremes[i] += 1;
                        if (num_extremes[i] == 1 || randomInt(num_extremes[i]) == 0)
                            is_extreme_kernel[i] = o;
                    }
                }
            }
        }
        break;
        // From the extreme_radius * |P| solutions with the best objective value
        // select extreme_radius / 1.5 * |P| solutions with GSS using hamming distance.
        case 7:
        {
            int k_1 = extreme_radius * population_size;
            std::vector<int> num_extremes(population_size);
            std::fill(num_extremes.begin(), num_extremes.end(), 0);
            std::vector<int> indices(population_size);
            setVectorToRandomOrdering(indices);
            // std::iota(indices.begin(), indices.end(), 0);

            for (int o = 0; o < number_of_objectives; ++o)
            {
                auto is_idx_a_better_than_idx_b = [&o](int idx_a, int idx_b)
                {
                    return is_objective_a_better(objective_values[idx_a][o], objective_values[idx_b][o], o);
                };

                std::nth_element(indices.begin(), indices.begin() + k_1, indices.end(),
                    is_idx_a_better_than_idx_b);
                // Assert sortedness.
                // for (int i_before = 0; i_before < k; i_before++)
                //     for (int i_after = k + 1; i_after < population_size; i_after++)
                //         assert(!is_objective_a_better(objective_values[indices[i_after]][o], objective_values[indices[i_before]][o], 0));
                
                int k_2 = extreme_radius / 1.5 * population_size; // std::min(k_1, static_cast<int>(std::ceil(4 * std::sqrt(population_size))));
                std::vector<int> selected(k_2);
                auto index_hamming_distance = [](int a, int b)
                {
                    return hammingDistanceInParameterSpace(population[a], population[b]);
                };
                performGreedySubsetScatteringWithDistance(k_1, k_2, index_hamming_distance, selected);

                int max_idx = *std::max_element(indices.begin(), indices.begin() + k_1, is_idx_a_better_than_idx_b);
                bool max_idx_among_selected = false;

                for (int selected_idx = 0; selected_idx < k_2; ++selected_idx)
                {
                    int indices_idx = selected[selected_idx];
                    if (indices_idx == max_idx)
                        max_idx_among_selected = true;
                    int i = indices[indices_idx];
                    num_extremes[i] += 1;
                    if (num_extremes[i] == 1 || randomInt(num_extremes[i]) == 0)
                    {
                        is_extreme_kernel[i] = o;
                    }
                }
                if (!max_idx_among_selected)
                {
                    int i = indices[max_idx];
                    num_extremes[i] += 1;
                    if (num_extremes[i] == 1 || randomInt(num_extremes[i]) == 0)
                    {
                        is_extreme_kernel[i] = o;
                    }
                }
            }

        }
        break;
        // Per rank:
        // From the extreme_radius * |R| solutions with the best objective value
        // select extreme_radius / 1.5 * |R| solutions with GSS using hamming distance.
        case 8:
        {
            determineRanks();
            std::vector<int> num_extremes(population_size);
            std::fill(num_extremes.begin(), num_extremes.end(), 0);
            std::vector<int> indices(population_size);

            int idx = 0;
            for (int r = 0; r < max_rank; ++r)
            {
                indices.clear();
                // Collect indices for current rank.
                for (; idx < population_size && rank[rank_order[idx]] == r; ++idx)
                {
                    int i = rank_order[idx];
                    indices.push_back(i);
                }
            
                for (int o = 0; o < number_of_objectives; ++o)
                {
                    auto is_idx_a_better_than_idx_b = [&o](int idx_a, int idx_b)
                    {
                        return is_objective_a_better(objective_values[idx_a][o], objective_values[idx_b][o], o);
                    };
                    int k_1 = std::ceil(extreme_radius * indices.size());

                    std::nth_element(indices.begin(), indices.begin() + k_1, indices.end(),
                        is_idx_a_better_than_idx_b);
                    // Assert sortedness.
                    // for (int i_before = 0; i_before < k; i_before++)
                    //     for (int i_after = k + 1; i_after < population_size; i_after++)
                    //         assert(!is_objective_a_better(objective_values[indices[i_after]][o], objective_values[indices[i_before]][o], 0));
                    
                    int k_2 = std::ceil(extreme_radius / 1.5 * indices.size()); // std::min(k_1, static_cast<int>(std::ceil(4 * std::sqrt(population_size))));
                    std::vector<int> selected(k_2);
                    auto index_hamming_distance = [](int a, int b)
                    {
                        return hammingDistanceInParameterSpace(population[a], population[b]);
                    };
                    performGreedySubsetScatteringWithDistance(k_1, k_2, index_hamming_distance, selected);

                    int max_idx = *std::max_element(indices.begin(), indices.begin() + k_1, is_idx_a_better_than_idx_b);
                    bool max_idx_among_selected = false;

                    for (int selected_idx = 0; selected_idx < k_2; ++selected_idx)
                    {
                        int indices_idx = selected[selected_idx];
                        if (indices_idx == max_idx)
                            max_idx_among_selected = true;
                        int i = indices[indices_idx];
                        num_extremes[i] += 1;
                        if (num_extremes[i] == 1 || randomInt(num_extremes[i]) == 0)
                        {
                            is_extreme_kernel[i] = o;
                        }
                    }
                    if (!max_idx_among_selected)
                    {
                        int i = indices[max_idx];
                        num_extremes[i] += 1;
                        if (num_extremes[i] == 1 || randomInt(num_extremes[i]) == 0)
                        {
                            is_extreme_kernel[i] = o;
                        }
                    }
                }
            }
        }
        break;
        // Select KNN of best
        // Case 10: also look at the solutions in between the best fitness value of the neighbor
        case 9: case 10: case 11:
        {
            std::vector<int> num_extremes(population_size);
            std::fill(num_extremes.begin(), num_extremes.end(), 0);
            std::vector<int> donor_indices;
            std::vector<int> indices(population_size);
            setVectorToRandomOrdering(indices);
            //std::iota(indices.begin(), indices.end(), 0);
            std::vector<char> seen(population_size);
            bool select_subset = extreme_kernel_mode == 11;
            int select_subset_redundancy = 3;

            for (int o = 0; o < number_of_objectives; ++o)
            {
                int best_idx = 0;
                double best_obj_value = (optimization[o] == MAXIMIZATION) ? -INFINITY : INFINITY;
                auto is_idx_a_better_than_idx_b = [&o](int idx_a, int idx_b)
                {
                    return is_objective_a_better(objective_values[idx_a][o], objective_values[idx_b][o], o);
                };

                for (int i = 0; i < population_size; ++i)
                {
                    double current_obj_value = objective_values[i][o];
                    if ( is_objective_a_better(current_obj_value, best_obj_value, o) )
                    {
                        best_idx = i;
                        best_obj_value = current_obj_value;   
                    }
                }

                // TODO: identify cluster if we want this to work with any clustering mode.
                int cluster_index = -1;
                int line_index = -1;
                determineMixingPoolForSolution(best_idx, cluster_index, line_index, donor_indices);
                double worst_obj_value_knn_best = (optimization[o] == MAXIMIZATION) ? INFINITY : -INFINITY;

                std::fill(seen.begin(), seen.end(), FALSE);

                is_extreme_kernel[best_idx] = o; 
                seen[best_idx] = TRUE;

                for (int idx = 0; idx < static_cast<int>(donor_indices.size()); ++idx)
                {
                    int i = donor_indices[idx];
                    seen[i] = TRUE;
                    num_extremes[i] += 1;
                    double current_obj_value = objective_values[i][o];
                    if (is_objective_a_better(worst_obj_value_knn_best, current_obj_value, o))
                        worst_obj_value_knn_best = current_obj_value;

                    if (
                        // is_objective_a_better(current_obj_value, min_obj_extreme, o) &&
                        !select_subset &&
                        (num_extremes[i] == 1 || randomInt(num_extremes[i]) == 0))
                    {
                        is_extreme_kernel[i] = o;
                    }
                }

                if (select_subset)
                {
                    std::vector<int> uncovered(number_of_parameters);
                    std::iota(uncovered.begin(), uncovered.end(), 0);
                    std::vector<int> cover_left(number_of_parameters);
                    std::fill(cover_left.begin(), cover_left.end(), select_subset_redundancy);
                    for (int idx = 0; idx < static_cast<int>(donor_indices.size()); ++idx)
                    {
                        int j = donor_indices[idx];
                        bool any_new_covered = false;
                        for (size_t o_idx = 0; o_idx < uncovered.size(); ++o_idx)
                        {
                            int o = uncovered[o_idx];
                            if (population[best_idx][o] != population[j][o])
                            {
                                // j covers o.
                                // - As such j covers something.
                                any_new_covered = true;
                                // - Reduce the number of cover_left for o
                                --cover_left[o];
                                // If we have hit zero covers left
                                if (cover_left[o] == 0)
                                {
                                    // - We should remove o from the uncovered set
                                    std::swap(uncovered[o_idx], uncovered[uncovered.size() - 1]);
                                    uncovered.resize(uncovered.size() - 1);
                                    // - And ensure we revisit the current index
                                    --o_idx;
                                }
                            }
                        }

                        if (any_new_covered)
                        {
                            is_extreme_kernel[j] = o;
                        }

                        if (uncovered.size() == 0)
                        {
                            // std::cout << "[pop:" << population_size << "][gen:" << number_of_generations << "][obj:" << o << "] Covered all for " << best_idx << " before end!\n";
                            break;
                        }
                    }
                }

                // Next steps are not required for mode 9.
                if (extreme_kernel_mode == 9) continue;

                // For each solution with a fitness better than worst_obj_value_knn_best
                // From best to worst.
                // We want to check if any solutions from their neighborhood have already been represented
                // or seen.
                // Otherwise they are part of a niche that the best solution's neighborhood did not cover.
                // And we should add this solution and their neighborhood.
                double lower_threshold = (worst_obj_value_knn_best + best_obj_value) / 2;
                int num_better_than_lower_threshold = 0;
                for (int i = 0; i < population_size; ++i)
                    if (is_objective_a_better(objective_values[i][o], lower_threshold, o)) 
                        ++num_better_than_lower_threshold;
                // Sort subset up to a point.
                std::nth_element(indices.begin(), indices.begin() + num_better_than_lower_threshold, indices.end(),
                    is_idx_a_better_than_idx_b);
                std::sort(indices.begin(), indices.begin() + num_better_than_lower_threshold, is_idx_a_better_than_idx_b);
                
                for (int idx = 0; idx < num_better_than_lower_threshold; ++idx)
                {
                    int i = indices[idx];
                    if (seen[i] == TRUE) continue;
                    // Determine mixing pool
                    determineMixingPoolForSolution(i, cluster_index, line_index, donor_indices);
                    // Determine whether we have seen any of our neighbors before.
                    int count_seen_before = 0;
                    for (int j: donor_indices)
                    {
                        if (seen[j]) ++count_seen_before;
                        seen[j] = TRUE;
                    }
                    seen[i] = TRUE;
                    // We have seen neighbors before: a similar neighborhood of better solutions has already
                    // been selected.
                    double ratio_seen_before = static_cast<double>(count_seen_before) / static_cast<double>(donor_indices.size());
                    if (ratio_seen_before >= extreme_radius) continue;
                    // std::cout << "i: " << i << " was not seen before and has " << ratio_seen_before << " overlap in neighbors."  << std::endl;
                    // std::cout << "best-worst objective values: " << best_obj_value << " - " << worst_obj_value_knn_best << "; i's objective value: " << objective_values[i][o] << std::endl;
                    // Otherwise we have found a solution most extreme which has not been seen
                    // in the neighborhood, nor has a neighborhood which has been seen itself.
                    // Likely a niche!
                    if (select_subset)
                    {
                        // std::vector<int> cover_count
                        std::vector<int> uncovered(number_of_parameters);
                        std::iota(uncovered.begin(), uncovered.end(), 0);
                        std::vector<int> cover_left(number_of_parameters);
                        std::fill(cover_left.begin(), cover_left.end(), select_subset_redundancy);
                        for (int idx = 0; idx < static_cast<int>(donor_indices.size()); ++idx)
                        {
                            int j = donor_indices[idx];
                            bool any_new_covered = false;
                            for (size_t o_idx = 0; o_idx < uncovered.size(); ++o_idx)
                            {
                                int o = uncovered[o_idx];
                                if (population[i][o] != population[j][o])
                                {
                                    // j covers o.
                                    // - As such j covers something.
                                    any_new_covered = true;
                                    // - Reduce the number of cover_left for o
                                    --cover_left[o];
                                    // If we have hit zero covers left
                                    if (cover_left[o] == 0)
                                    {
                                        // - We should remove o from the uncovered set
                                        std::swap(uncovered[o_idx], uncovered[uncovered.size() - 1]);
                                        uncovered.resize(uncovered.size() - 1);
                                        // - And ensure we revisit the current index
                                        --o_idx;
                                    }
                                }
                            }

                            if (any_new_covered)
                            {
                                is_extreme_kernel[j] = o;
                            }

                            if (uncovered.size() == 0)
                            {
                                // std::cout << "[pop:" << population_size << "][gen:" << number_of_generations << "][obj:" << o << "] Covered all for " << i << " before end!\n";
                                break;
                            }
                        }
                    }
                    else
                    for (int idx = 0; idx < static_cast<int>(donor_indices.size()); ++idx)
                    {
                        int j = donor_indices[idx];
                        // double current_obj_value = objective_values[j][o];
                        num_extremes[j] += 1;
                        if (
                            // is_objective_a_better(current_obj_value, min_obj_extreme, o) &&
                            // idx <= std::ceil(2 * std::sqrt(population_size)) &&
                            (num_extremes[j] == 1 || randomInt(num_extremes[j]) == 0))
                        {
                            is_extreme_kernel[j] = o;
                        }
                    }
                }
            }
        }
        break;
    }
    
}

void initializeClusters()
{
    int i;
    lt                            = (int ***) Malloc( number_of_mixing_components*sizeof( int ** ) );
    lt_length                     = (int *) Malloc( number_of_mixing_components*sizeof( int ) );
    lt_number_of_indices          = (int **) Malloc( number_of_mixing_components*sizeof( int *) );
    for( i = 0; i < number_of_mixing_components; i++)
    {
        lt[i]                     = NULL;
        lt_number_of_indices[i]   = NULL;
        lt_length[i]              = 0;
    }

    which_extreme                 = (int*)Malloc(number_of_mixing_components*sizeof(int));

    objective_means_scaled        = (double **) Malloc( number_of_mixing_components*sizeof( double * ) );    
    for( i = 0; i < number_of_mixing_components; i++ )
        objective_means_scaled[i] = (double *) Malloc( number_of_objectives*sizeof( double ) );
}

void ezilaitiniClusters()
{
    int i, j;

    if(lt == NULL)
        return;

    for( i = 0; i < number_of_mixing_components; i++ )
    {
        if( lt[i] != NULL )
        {
            for( j = 0; j < lt_length[i]; j++ )
                free( lt[i][j] );
            free( lt[i] );
            free( lt_number_of_indices[i] );
        }
    }

    free( lt ); lt = NULL;
    free( lt_length );
    free( lt_number_of_indices );

    free(which_extreme);

    for(i = 0; i < number_of_mixing_components; i++)
        free(objective_means_scaled[i]);
    free( objective_means_scaled );

    is_extreme_kernel.resize(0);
    ranks_determined = false;

    if (orig_number_of_mixing_components < 0)
    {
        // End the override: back to normal!
        number_of_mixing_components = orig_number_of_mixing_components;
        orig_number_of_mixing_components = -1;
    }
}

void improveCurrentPopulation( int mode )
{
    int     i, j, k, j_min, cluster_index, objective_index, number_of_cluster,
            *sum_cluster, *clusters ;
    double *objective_values_scaled,
          distance, distance_smallest;

    offspring_size                  = population_size;
    offspring                       = (char**)Malloc(offspring_size*sizeof(char*));
    objective_values_offspring      = (double**)Malloc(offspring_size*sizeof(double*));
    constraint_values_offspring     = (double*)Malloc(offspring_size*sizeof(double));
    solution_metadata_offspring     = (void**)Malloc(offspring_size*sizeof(void*));

    for(i = 0; i < offspring_size; i++)
    {
        offspring[i]                = (char*)Malloc(number_of_parameters*sizeof(char));
        objective_values_offspring[i]  = (double*)Malloc(number_of_objectives*sizeof(double));
        initSolutionMetadata(solution_metadata_offspring[i]);
    }

    objective_values_scaled = (double *) Malloc( number_of_objectives*sizeof( double ) );
    sum_cluster = (int*)Malloc(number_of_mixing_components*sizeof(int));

    for(i = 0; i < number_of_mixing_components; i++)
        sum_cluster[i] = 0;

    if (hv_density_correction)
        updateCurrentKernelDensityEstimate();

    bool mode_uses_scalarization = (mode == -1);

    int* scalarization_assignment = NULL;
    double* scalarization_ranges = NULL;
    double* scalarization_ideal = NULL;
    if (mode_uses_scalarization)
    {
        // If we try to use a this particular scalarized variant without this being true, things will go wrong:
        // For example tsch_weight_vectors is NULL (and using it will probably lead to a segfault!)
        assert(use_scalarization);
        
        scalarization_assignment = scalarizedAssignWeightVectorsEvenly(
            number_of_objectives, optimization, population_size, objective_values, population_size, tsch_weight_vectors,
            scalarization_ranges, scalarization_ideal);
    }
    double** pseudo_objective_values = NULL;
    if (mode == -2 || mode == -3)
    {
        // Create copy
        pseudo_objective_values = (double**) Malloc(population_size * sizeof(double*));
        std::copy(objective_values, objective_values + population_size, pseudo_objective_values);
    }

    elitist_archive_front_changed = FALSE;
    for( i = 0; i < population_size; i++ )
    {
        number_of_cluster = 0;
        clusters = (int*)Malloc(number_of_mixing_components*sizeof(int));
        for(j = 0; j < number_of_mixing_components; j++)
        {
            for (k = 0; k < population_cluster_sizes[j]; k++)
            {
                if(population_indices_of_cluster_members[j][k] == i)
                {
                    clusters[number_of_cluster] = j;
                    number_of_cluster++;
                    break;
                }
            }
        }
        // If there exists a valid cluster assignment, use it.
        if (static_cast<int>(mixture_assignment.size()) == population_size)
        {
            assert(number_of_mixing_components == population_size);
            cluster_index = mixture_assignment[i];
        }
        // If we found clusters this element is in, select one of those.
        else if(number_of_cluster > 0)
            cluster_index = clusters[randomInt(number_of_cluster)];
        // Otherwise resolve the missing cluster by taking the nearest one according to some metric.
        else
        {
            // std::cout << "No cluster for population member with index " << i << " at evaluation point " << number_of_evaluations << ".\n";
            int resolve_missing_cluster_by = 0;
            if (!use_original_clustering_with_domination_iteration && (approach_mode == 1 || approach_mode == 2 || approach_mode == 3))
                resolve_missing_cluster_by = 1;

            if (resolve_missing_cluster_by == 0)
            {
                // Resolve missing cluster by finding the nearest cluster.

                for( j = 0; j < number_of_objectives; j++ )
                    objective_values_scaled[j] = objective_values[i][j]/objective_ranges[j];

                distance_smallest = -1;
                j_min = -1;
                for( j = 0; j < number_of_mixing_components; j++ )
                {
                    distance = distanceEuclidean( objective_values_scaled, objective_means_scaled[j], number_of_objectives );
                    if( (distance_smallest < 0) || (distance < distance_smallest) )
                    {
                        j_min = j;
                        distance_smallest  = distance;
                    }
                
                }

                cluster_index = j_min;
            }
            else if (resolve_missing_cluster_by == 1)
            {
                // Simply pick the nearest line!.

                cluster_index = nearest_line[i];
            }
        }

        sum_cluster[cluster_index]++;

        if (static_cast<int>(is_extreme_kernel.size()) == population_size &&
            is_extreme_kernel[i] != -1)
        {
            objective_index = is_extreme_kernel[i];
            performSingleObjectiveGenepoolOptimalMixing(
                cluster_index, objective_index, i, 
                population[i], objective_values[i], constraint_values[i], solution_metadata[i],  
                offspring[i], objective_values_offspring[i], &(constraint_values_offspring[i]), solution_metadata_offspring[i]);
        }
        else if (mode == 1)
        {
            if (log_mixing_invocations)
                std::cout << "Starting multiobjective line mixing on " << i << " using line " << assigned_line[i] << " at " << number_of_evaluations << " evaluations." << '\n';
            
            performLineMultiObjectiveGenepoolOptimalMixing(
                cluster_index, assigned_line[i], i,
                population[i], objective_values[i], constraint_values[i], solution_metadata[i],
                offspring[i], objective_values_offspring[i], &(constraint_values_offspring[i]), solution_metadata_offspring[i]);
        }
        else if (mode == 2)
        {
            
            if (log_mixing_invocations)
                std::cout << "Starting pure (incl. single objective clusters) multiobjective mixing on " << i << " at " << number_of_evaluations << " evaluations." << '\n';
            performMultiObjectiveGenepoolOptimalMixing(
                cluster_index, i,
                population[i], objective_values[i], constraint_values[i], solution_metadata[i],
                offspring[i], objective_values_offspring[i], &(constraint_values_offspring[i]), solution_metadata_offspring[i]);
        }
        else if (mode == 3)
        {
            
            if (log_mixing_invocations)
                std::cout << "Starting multiobjective line mixing on " << i << " in kernel style " << kernel_improvement_mode << " at " << number_of_evaluations << " evaluations." << '\n';
            performLineMultiObjectiveGenepoolOptimalMixing( 
                cluster_index, kernel_improvement_mode, i, 
                population[i], objective_values[i], constraint_values[i], solution_metadata[i],
                offspring[i], objective_values_offspring[i], &(constraint_values_offspring[i]), solution_metadata_offspring[i]);
        }
        else if (mode == -1) // && which_extreme[cluster_index] == -1
        {
            int i_w = scalarization_assignment[i];
            // Scalarized mixing!
            if (log_mixing_invocations)
                std::cout << "Starting scalarized mixing on " << i << " with weight vector " << i_w << " at "
                          << number_of_evaluations << " evaluations." << '\n';
            performTschebysheffObjectiveGenepoolOptimalMixing(
                cluster_index, i,
                population[i],objective_values[i], constraint_values[i], solution_metadata[i],
                offspring[i], objective_values_offspring[i], &(constraint_values_offspring[i]), solution_metadata_offspring[i],
                tsch_weight_vectors[i_w], scalarization_ideal, scalarization_ranges);
        }
        else if (mode == -2)
        {
            if (log_mixing_invocations)
                std::cout << "Starting RankHV mixing on " << i << " at "
                          << number_of_evaluations << " evaluations." << '\n';
            performRankHVObjectiveGenepoolOptimalMixing(cluster_index, i, pseudo_objective_values, population[i],objective_values[i], constraint_values[i], solution_metadata[i],
                offspring[i], objective_values_offspring[i], &(constraint_values_offspring[i]), solution_metadata_offspring[i]);
        }
        else if (mode == -3)
        {
            if (log_mixing_invocations)
                std::cout << "Starting UHV mixing on " << i << " at "
                          << number_of_evaluations << " evaluations." << '\n';
            performUHVIObjectiveGenepoolOptimalMixing(cluster_index, i, pseudo_objective_values, population[i],objective_values[i], constraint_values[i], solution_metadata[i],
                offspring[i], objective_values_offspring[i], &(constraint_values_offspring[i]), solution_metadata_offspring[i]);
        }
        else
        {
            if(which_extreme[cluster_index] == -1)
            {
                if (log_mixing_invocations)
                    std::cout << "Starting multiobjective mixing on " << i << " at " << number_of_evaluations << " evaluations." << '\n';
                performMultiObjectiveGenepoolOptimalMixing( 
                    cluster_index, i,
                    population[i], objective_values[i], constraint_values[i], solution_metadata[i],
                    offspring[i], objective_values_offspring[i], &(constraint_values_offspring[i]), solution_metadata_offspring[i]);
            }
            else
            {
                objective_index = which_extreme[cluster_index];
                
                if (log_mixing_invocations)
                {
                    std::cout << "Starting single objective mixing on " << i;
                    for (int o = 0; o < number_of_objectives; ++o)
                        std::cout << (o == 0 ? " (" : ", ") << objective_values[i][o];
                    std::cout << ") ";
                    std::cout << " for objective " << objective_index << " at " << number_of_evaluations << " evaluations." << '\n';
                }
                
                performSingleObjectiveGenepoolOptimalMixing(
                    cluster_index, objective_index, i, 
                    population[i], objective_values[i], constraint_values[i], solution_metadata[i],  
                    offspring[i], objective_values_offspring[i], &(constraint_values_offspring[i]), solution_metadata_offspring[i]);
            }
        }
        free(clusters);
    }

    free( objective_values_scaled ); free( sum_cluster );

    if (pseudo_objective_values != NULL)
    {
        free(pseudo_objective_values);
    }

    if (mode_uses_scalarization)
    {
        free(scalarization_assignment);
        free(scalarization_ranges);
        free(scalarization_ideal);
    }

    if(!elitist_archive_front_changed)
        t_NIS++;
    else
        t_NIS = 0;
}

void copyValuesFromDonorToOffspring(char *solution, char *donor, int cluster_index, int linkage_group_index)
{
    int i, parameter_index;
    for (i = 0; i < lt_number_of_indices[cluster_index][linkage_group_index]; i++)
    {
        parameter_index = lt[cluster_index][linkage_group_index][i];
        solution[parameter_index] = donor[parameter_index];    
    }
}

void copyFromAToB(char *solution_a, double *obj_a, double con_a, void* &metadata_a, char *solution_b, double *obj_b, double *con_b, void* &metadata_b)
{
    int i;
    for (i = 0; i < number_of_parameters; i++)
        solution_b[i] = solution_a[i];
    for (i = 0; i < number_of_objectives; i++)
        obj_b[i] = obj_a[i];
    *con_b = con_a;
    copySolutionMetadata(metadata_a, metadata_b);
}

void copyFromAToBSubset(char *solution_a, double *obj_a, double con_a, void* &metadata_a, char *solution_b, double *obj_b, double *con_b, void* &metadata_b, int* subset, int subset_size)
{
    int i;
    for (i = 0; i < subset_size; i++)
        solution_b[subset[i]] = solution_a[subset[i]];
    for (i = 0; i < number_of_objectives; i++)
        obj_b[i] = obj_a[i];
    *con_b = con_a;
    copySolutionMetadata(metadata_a, metadata_b);
}


void mutateSolution(char *solution, int lt_factor_index, int cluster_index)
{
    double mutation_rate, prob;
    int i, parameter_index;

    if(use_pre_mutation == FALSE && use_pre_adaptive_mutation == FALSE)
        return;

    mutation_rate = 0.0;
    if(use_pre_mutation == TRUE)
        mutation_rate = 1.0/((double)number_of_parameters);
    else if(use_pre_adaptive_mutation == TRUE)
        mutation_rate = 1.0/((double)lt_number_of_indices[cluster_index][lt_factor_index]);

    
    for(i = 0; i < lt_number_of_indices[cluster_index][lt_factor_index]; i++)
    {
        prob = randomRealUniform01();
        if(prob < mutation_rate)
        {
            parameter_index = lt[cluster_index][lt_factor_index][i];
            if(solution[parameter_index] == 0) 
                solution[parameter_index] = 1;
            else
                solution[parameter_index] = 0;
        }
    }
    
}

double projectedLineObjectiveDistance(double* origin, double* direction, double* point)
{
    double r = 0.0;
    for (int o = 0; o < number_of_objectives; ++o)
    {
        double dot_o = (point[o] - origin[o]) * direction[o];
        // Note: Also normalize objectives here, when relevant.
        if (normalize_objectives_for_lines)
            dot_o /= objective_ranges[o];
        r += dot_o;
    }
    return r;
}

struct FOS
{
    int number_of_subsets;
    int* number_of_elements_per_subset;
    int** subsets;
};

double computeFOSDistance(char *a, char *b, FOS subsets)
{
    int number_of_matches = 0;

    for (int subset_idx = 0; subset_idx < subsets.number_of_subsets; ++subset_idx)
    {
        int elements_in_subset = subsets.number_of_elements_per_subset[subset_idx];
        int* subset = subsets.subsets[subset_idx];

        bool matches_all = true;

        for (int el_idx = 0; el_idx < elements_in_subset; ++el_idx)
        {
            int el = subset[el_idx];
            if (a[el] != b[el])
            {
                matches_all = false;
                break;
            }
        }

        if (matches_all)
            number_of_matches += 1;
    }

    return number_of_matches;
}

double projectedLineObjectiveDistance(int line_idx, double* point)
{
    double r = 0.0;
    for (int o = 0; o < number_of_objectives; ++o)
    {
        double dot_o = (point[o] - lines[line_idx].second[o]) * lines_direction_normalized[line_idx][o];
        // Note: Also normalize objectives here, when relevant.
        if (normalize_objectives_for_lines)
            dot_o /= objective_ranges[o];
        r += dot_o;
    }
    return r;
}

int determineKnnK(int knn_k_mode)
{
    // Positive integers are == to the number of neighbors.
    if (knn_k_mode > 0) return knn_k_mode;
    
    if (knn_k_mode == 0) return std::ceil(std::log2(population_size));
    if (knn_k_mode == -1) return std::ceil(std::sqrt(population_size));
    if (knn_k_mode == -2) return std::ceil(2 * static_cast<double>(population_size) / static_cast<double>(number_of_mixing_components));
    if (knn_k_mode == -3) return std::ceil(2 * std::sqrt(population_size));

    std::cerr << "Invalid KNN k mode provided." << std::endl;
    std::exit(1);
}

void computeNormalizedDirectionVector(std::vector<double> &u, double* from, double* to, double* ranges, int clip = 1)
{
    u.resize(number_of_objectives);
    
    double length = 0.0;
    for (int d = 0; d < number_of_objectives; ++d)
    {
        // Use backup as reference.
        double delta = (to[d] - from[d]);
        if (ranges[d] != 0)
            delta /= ranges[d];
        // Cap by nadir: only allow negative deltas for minimization (lower is better)
        // and positive deltas for maximization (higher is better)
        if (clip == 1)
        {
            if (optimization[d] == MINIMIZATION)
                delta = std::min(delta, 0.0);
            else
                delta = std::max(delta, 0.0);
        }

        // Use new point as reference.
        // double delta = (obj[d] - nadir_point[d]) / objective_ranges[d];
        u[d] = delta;

        length += delta * delta;
    }
    // Normalization, not required for dot product, but probably required
    // When computing the distance to a line.
    length = std::sqrt(length);
    if (length > 0)
        for (int d = 0; d < number_of_objectives; ++d)
        {
            u[d] /= length;
        }
}

double computeDistanceOfKind(int distance_kind, int from, int to, std::vector<double> &direction_from, std::optional<FOS> &fos, double *ranges)
{
    double negate_multiplier = (distance_kind < 0) ? -1.0 : 1.0;
    distance_kind = std::abs(distance_kind);

    if (distance_kind == 1)
    {
        return negate_multiplier * distanceEuclidean(objective_values[from], objective_values[to], number_of_objectives, ranges);
    }
    else if (distance_kind == 2)
    {
        return negate_multiplier * euclideanDistanceToRay(nadir_point.data(), direction_from.data(), objective_values[to],
                                      number_of_objectives, ranges);
    }
    else if (distance_kind == 3)
    {
        return negate_multiplier * hammingDistanceInParameterSpace(population[from], population[to]);
    }
    else if (distance_kind == 4)
    {
        assert(fos.has_value());
        return negate_multiplier * computeFOSDistance(population[from], population[to], fos.value());
    }
    else
    {
        std::cerr << "Unknown distance_kind " << distance_kind << "." << std::endl;
        std::exit(1);
    }
}

/**
 * Generate a vector of pairs (distance, index) where distance is the distance from solution_index to index.
 *
 * Distance kind can be one of the following:
 * 1 - Euclidean distance (Objective Space)
 * 2 - Line distance (Objective Space)
 * 3 - Hamming distance (Parameter Space)
 * 4 - FOS distance (Parameter Space)
 *
 * Notes:
 * - index != solution_index
 * - fos is required if distance_kind = 4
 */
std::vector<std::pair<double, int>> createDistanceIndexVector(int solution_index, int distance_kind, std::optional<FOS> &fos)
{
    std::vector<std::pair<double, int>> distance_index_pairs(population_size - 1);
    std::vector<double> u;
    double *ranges = objective_ranges;

    // Determine rescaled direction vector if neccesary.
    if (distance_kind == 2)
    {
        computeNormalizedDirectionVector(u, nadir_point.data(), objective_values[solution_index], ranges);
    }

    // Compute distances.
    int pair_idx = 0;
    for (int j = 0; j < population_size; ++j)
    {
        if (j == solution_index)
            continue;

        double distance = computeDistanceOfKind(distance_kind, solution_index, j, u, fos, ranges);

        distance_index_pairs[pair_idx] = std::make_pair(distance, j);
        ++pair_idx;
    }

    return distance_index_pairs;
}

void updateDistanceIndexVector(int solution_index, int distance_kind, std::vector<std::pair<double, int>> &distance_index_pairs, std::optional<FOS> &fos)
{
    std::vector<double> u;
    double *ranges = objective_ranges;

    // Determine rescaled direction vector if neccesary.
    if (distance_kind == 2)
    {
        computeNormalizedDirectionVector(u, nadir_point.data(), objective_values[solution_index], ranges);
    }

    // Update vector
    int number_of_distance_index_pairs = distance_index_pairs.size();
    for (int j = 0; j < number_of_distance_index_pairs; ++j)
    {
        int idx = distance_index_pairs[j].second;
        double distance = computeDistanceOfKind(distance_kind, solution_index, idx, u, fos, ranges);
        distance_index_pairs[j] = std::make_pair(distance, idx);
    }

}

void determineLineDirection(int line_idx_x, double *obj_backup, double *obj, double *&line_origin,
                            double *&line_direction, std::vector<double> &u)
{
    if (line_idx_x >= 0)
    {
        line_origin = lines[line_idx_x].second.data();
        line_direction = lines_direction_normalized[line_idx_x].data();
    }
    else
    {
        // Negative index means kernel style direction.
        u.resize(number_of_objectives);
        if (line_idx_x == -1)
            // Default - always use original point.
            std::copy(obj_backup, obj_backup + number_of_objectives, u.begin());
        else if (line_idx_x == -2)
            // Use new point
            std::copy(obj, obj + number_of_objectives, u.begin());
        else
        {
            std::cerr << "Invalid Kernel Improvement Mode." << std::endl;
            exit(1);
        }

        computeNormalizedDirectionVector(u, nadir_point.data(), u.data(), objective_ranges);

        line_origin = nadir_point.data();
        line_direction = u.data();
    }
}

std::vector<int> solutionNeighborhoodRanks;

std::vector<std::pair<double, int>> createDistanceIndexVector(int solution_index, int distance_kind, std::optional<FOS> &fos);
std::vector<std::vector<int>> predeterminedMixingPools;

void performPrecomputationForMixingPool()
{
    if (mixing_pool_mode == 3 || mixing_pool_mode == 4)
    {
        solutionNeighborhoodRanks.resize(population_size * population_size);
        std::fill(solutionNeighborhoodRanks.begin(), solutionNeighborhoodRanks.end(), 0);

        for (int i = 0; i < population_size; ++i)
        {
            // TODO: Deteremine fos when using inverted distance
            // Only used with distance_kind = 4 right now.
            std::optional<FOS> fos;
            // Create a vector of distance index pairs.
            std::vector<std::pair<double, int>> distance_index_pairs = createDistanceIndexVector(i, mixing_pool_mode - 2, fos);
            // And sort it.
            std::sort(distance_index_pairs.begin(), distance_index_pairs.end());
            // Copy over the resulting ranks.
            // 
            int dip_size = distance_index_pairs.size();
            for (int r = 0; r < dip_size; ++r)
            {
                auto el = distance_index_pairs[r];
                // Write output such that each element is a 'row'.
                solutionNeighborhoodRanks[el.second * population_size + i] = r;
            }
        }
    }
    else if (mixing_pool_mode == 15 || mixing_pool_mode == 16)
    {
        std::optional<FOS> fos;
        bool deduplicate_mixing_pool = mixing_pool_mode == 16;
        // Hamming Symmetric KNN.
        int k = determineKnnK(mixing_pool_knn_k_mode);
        
        // Reset mixing pools
        predeterminedMixingPools.clear();
        predeterminedMixingPools.resize(population_size);

        for (int solution_index = 0; solution_index < population_size; ++solution_index)
        {
            std::vector<std::pair<double, int>> distance_index_pairs = createDistanceIndexVector(solution_index, 3, fos);

            if (ignore_identical_solutions_for_knn)
            {
                // Note: It may happen that this removes ALL solutions.
                char *self_start = population[solution_index];
                auto new_end = std::remove_if(distance_index_pairs.begin(), distance_index_pairs.end(),
                    [self_start](std::pair<double, int> &a)
                    {
                        return std::equal(self_start, self_start + number_of_parameters, population[a.second]);
                    });
                distance_index_pairs.resize(std::distance(distance_index_pairs.begin(), new_end));
            }

            if (k < static_cast<int>(distance_index_pairs.size()))
            {
                std::nth_element(distance_index_pairs.begin(), distance_index_pairs.begin() + k, distance_index_pairs.end());
                // assert(distance_index_pairs[k - 1] <= distance_index_pairs[k]);
                // assert(distance_index_pairs[k] <= distance_index_pairs[k + 1]);
                distance_index_pairs.resize(k);
            }

            if (deduplicate_mixing_pool)
            {
                std::sort(distance_index_pairs.begin(), distance_index_pairs.end(), 
                    [](std::pair<double, int> a, std::pair<double, int> b)
                    {
                        char *p_a = population[a.second];
                        char *p_b = population[b.second];
                        return std::lexicographical_compare(
                            p_a, p_a + number_of_parameters,
                            p_b, p_b + number_of_parameters
                        );
                    });
                auto new_end = std::unique(distance_index_pairs.begin(), distance_index_pairs.end(),
                    [](std::pair<double, int> a, std::pair<double, int> b)
                    {
                        char *p_a = population[a.second];
                        char *p_b = population[b.second];
                        return std::equal(
                            p_a, p_a + number_of_parameters,
                            p_b, p_b + number_of_parameters
                        );
                    });
                distance_index_pairs.resize(std::distance(distance_index_pairs.begin(), new_end));
            }

            k = distance_index_pairs.size();
            std::vector<int> &donor_indices = predeterminedMixingPools[solution_index];
            int current_size = donor_indices.size();
            donor_indices.resize(current_size + k);
            for (int j_idx = 0; j_idx < k; ++j_idx)
            {
                // Add KNN to solution_index's mixing pool
                donor_indices[current_size + j_idx] = distance_index_pairs[j_idx].second;
                // Additionally, add current solution_index to the KNN's mixing pool.
                predeterminedMixingPools[distance_index_pairs[j_idx].second].push_back(solution_index);
            }
            
        }

        // Deduplicate indices in each mixing pool.
        std::vector<char> seen(population_size);
        for (int solution_index = 0; solution_index < population_size; ++solution_index)
        {
            std::fill(seen.begin(), seen.end(), false);
            std::vector<int> &donor_indices = predeterminedMixingPools[solution_index];
            auto new_end = std::remove_if(donor_indices.begin(), donor_indices.end(), 
                [&seen](int i)
                {
                    bool was_seen_before = seen[i];
                    seen[i] = true;
                    return was_seen_before;
                });
            donor_indices.erase(new_end, donor_indices.end());
        }


    }
    // mergeSort()
}

/**
 * Determine which solutions can be mixed with, depending on the mixing pool mode.
 */
void determineMixingPoolForSolution(int solution_index, int cluster_index, int /* line_index */, std::vector<int> &donor_indices)
{
    
    std::optional<FOS> fos;
    if (cluster_index >= 0)
    {
        fos = FOS {
            lt_length[cluster_index],
            lt_number_of_indices[cluster_index],
            lt[cluster_index]
        };
    }

    if (mixing_pool_mode == -2)
    {
        donor_indices.resize(population_size);
        std::iota(donor_indices.begin(), donor_indices.end(), 0);
    }
    else if (mixing_pool_mode == -1)
    {
        // Using cluster by solution index, indices are contained within cluster.

        // If this is not true, we'll end up indexing out of bounds
        // i.e. this mixing pool mode is invalid for the current approach.
        assert(number_of_mixing_components == population_size);

        int s = population_cluster_sizes[solution_index];
        donor_indices.resize(s);
        std::copy(population_indices_of_cluster_members[solution_index], 
                  population_indices_of_cluster_members[solution_index] + s,
                  donor_indices.begin());
    }
    else if (mixing_pool_mode == 0)
    {
        // Using cluster, indices are contained within cluster.
        int s = population_cluster_sizes[cluster_index];
        donor_indices.resize(s);
        std::copy(population_indices_of_cluster_members[cluster_index], 
                  population_indices_of_cluster_members[cluster_index] + s,
                  donor_indices.begin());
    }
    else if (mixing_pool_mode == 1 || mixing_pool_mode == 2)
    {
        // Euclidean KNN or Line KNN.
        int k = determineKnnK(mixing_pool_knn_k_mode);

        std::vector<std::pair<double, int>> distance_index_pairs = createDistanceIndexVector(solution_index, mixing_pool_mode, fos);
        
        if (ignore_identical_solutions_for_knn)
        {
            char *self_start = population[solution_index];
            auto new_end = std::remove_if(distance_index_pairs.begin(), distance_index_pairs.end(),
                [self_start](std::pair<double, int> &a)
                {
                    return std::equal(self_start, self_start + number_of_parameters, population[a.second]);
                });
            distance_index_pairs.resize(std::distance(distance_index_pairs.begin(), new_end));
        }

        if (k < static_cast<int>(distance_index_pairs.size()))
        {
            std::nth_element(distance_index_pairs.begin(), distance_index_pairs.begin() + k, distance_index_pairs.end());
            // assert(distance_index_pairs[k - 1] <= distance_index_pairs[k]);
            // assert(distance_index_pairs[k] <= distance_index_pairs[k + 1]);
            distance_index_pairs.resize(k);
        }

        k = distance_index_pairs.size();
        donor_indices.resize(k);
        for (int j_idx = 0; j_idx < k; ++j_idx)
        {
            donor_indices[j_idx] = distance_index_pairs[j_idx].second;
        }
    }
    else if (mixing_pool_mode == 3 || mixing_pool_mode == 4)
    {
        // Euclidean KNN or Line KNN.
        int k = determineKnnK(mixing_pool_knn_k_mode);
        // Check!
        assert(static_cast<int>(solutionNeighborhoodRanks.size()) == population_size * population_size);

        std::vector<std::pair<int, int>> rank_index_pairs(population_size - 1);
        int pair_idx = 0;
        for (int j = 0; j < population_size; ++j)
        {
            if (j == solution_index) continue;
            rank_index_pairs[pair_idx] = std::make_pair(solutionNeighborhoodRanks[j], j);            
            ++pair_idx;
        }

        if (k < population_size - 1)
        {
            std::nth_element(rank_index_pairs.begin(), rank_index_pairs.begin() + k, rank_index_pairs.end());
            // assert(rank_index_pairs[k - 1] <= rank_index_pairs[k]);
            // assert(rank_index_pairs[k] <= rank_index_pairs[k + 1]);
            rank_index_pairs.resize(k);
        }
        k = rank_index_pairs.size();

        donor_indices.resize(k);
        for (int j_idx = 0; j_idx < k; ++j_idx)
        {
            donor_indices[j_idx] = rank_index_pairs[j_idx].second;
        }
    }
    else if (
        mixing_pool_mode == 5 || mixing_pool_mode == 6 || 
        mixing_pool_mode == 7 || mixing_pool_mode == 8 ||
        mixing_pool_mode == 10 || mixing_pool_mode == 11 ||
        mixing_pool_mode == 12 || mixing_pool_mode == 13)
    {
        // Initialize to silence maybe uninitialized:
        // all six of the values above are covered, as such
        // not neccesary.
        int distance_kind_a = 1;
        int distance_kind_b = -3;
        switch (mixing_pool_mode)
        {
            case 5:
                distance_kind_a = 1;
                distance_kind_b = -3;
                break;
            case 6:
                distance_kind_a = 2;
                distance_kind_b = -3;
                break;
            case 7:
                distance_kind_a = 1;
                distance_kind_b = -4;
                break;
            case 8:
                distance_kind_a = 2;
                distance_kind_b = -4;
                break;
            case 10:
                distance_kind_a = 1;
                distance_kind_b = 3;
                break;
            case 11:
                distance_kind_a = 2;
                distance_kind_b = 3;
                break;
            case 12:
                distance_kind_b = 3;
                distance_kind_a = 1;
                break;
            case 13:
                distance_kind_b = 3;
                distance_kind_a = 2;
                break;
        }

         // Euclidean KNN or Line KNN.
        int k = determineKnnK(mixing_pool_knn_k_mode);

        std::vector<std::pair<double, int>> distance_index_pairs = createDistanceIndexVector(solution_index, distance_kind_a, fos);
        
        if (ignore_identical_solutions_for_knn)
        {
            // Note: It may happen that this removes ALL solutions.
            char *self_start = population[solution_index];
            auto new_end = std::remove_if(distance_index_pairs.begin(), distance_index_pairs.end(),
                [self_start](std::pair<double, int> &a)
                {
                    return std::equal(self_start, self_start + number_of_parameters, population[a.second]);
                });
            distance_index_pairs.resize(std::distance(distance_index_pairs.begin(), new_end));
        }

        if (2 * k < static_cast<int>(distance_index_pairs.size()))
        {
            std::nth_element(distance_index_pairs.begin(), distance_index_pairs.begin() + 2 * k, distance_index_pairs.end());
            // assert(distance_index_pairs[k - 1] <= distance_index_pairs[k]);
            // assert(distance_index_pairs[k] <= distance_index_pairs[k + 1]);
            distance_index_pairs.resize(k);
        }

        // Compute distances of secondary metric: Hamming Distance
        // This metric should be maximized instead.
        updateDistanceIndexVector(solution_index, distance_kind_b, distance_index_pairs, fos);

        // And trim again.
        if (k < static_cast<int>(distance_index_pairs.size()))
        {
            std::nth_element(distance_index_pairs.begin(), distance_index_pairs.begin() + k, distance_index_pairs.end());
            // assert(distance_index_pairs[k - 1] <= distance_index_pairs[k]);
            // assert(distance_index_pairs[k] <= distance_index_pairs[k + 1]);
            distance_index_pairs.resize(k);
        }

        k = distance_index_pairs.size();
        donor_indices.resize(k);
        for (int j_idx = 0; j_idx < k; ++j_idx)
        {
            donor_indices[j_idx] = distance_index_pairs[j_idx].second;
        }
    }
    else if (mixing_pool_mode == 9 || mixing_pool_mode == 14)
    {
        bool deduplicate_mixing_pool = mixing_pool_mode == 14;
        // Hamming KNN.
        int k = determineKnnK(mixing_pool_knn_k_mode);

        std::vector<std::pair<double, int>> distance_index_pairs = createDistanceIndexVector(solution_index, 3, fos);

        if (ignore_identical_solutions_for_knn)
        {
            // Note: It may happen that this removes ALL solutions.
            char *self_start = population[solution_index];
            auto new_end = std::remove_if(distance_index_pairs.begin(), distance_index_pairs.end(),
                [self_start](std::pair<double, int> &a)
                {
                    return std::equal(self_start, self_start + number_of_parameters, population[a.second]);
                });
            distance_index_pairs.resize(std::distance(distance_index_pairs.begin(), new_end));
        }

        if (k < static_cast<int>(distance_index_pairs.size()))
        {
            std::nth_element(distance_index_pairs.begin(), distance_index_pairs.begin() + k, distance_index_pairs.end());
            // assert(distance_index_pairs[k - 1] <= distance_index_pairs[k]);
            // assert(distance_index_pairs[k] <= distance_index_pairs[k + 1]);
            distance_index_pairs.resize(k);
        }

        if (deduplicate_mixing_pool)
        {
            std::sort(distance_index_pairs.begin(), distance_index_pairs.end(), 
                [](std::pair<double, int> a, std::pair<double, int> b)
                {
                    char *p_a = population[a.second];
                    char *p_b = population[b.second];
                    return std::lexicographical_compare(
                        p_a, p_a + number_of_parameters,
                        p_b, p_b + number_of_parameters
                    );
                });
            auto new_end = std::unique(distance_index_pairs.begin(), distance_index_pairs.end(),
                [](std::pair<double, int> a, std::pair<double, int> b)
                {
                    char *p_a = population[a.second];
                    char *p_b = population[b.second];
                    return std::equal(
                        p_a, p_a + number_of_parameters,
                        p_b, p_b + number_of_parameters
                    );
                });
            distance_index_pairs.resize(std::distance(distance_index_pairs.begin(), new_end));
        }

        k = distance_index_pairs.size();
        donor_indices.resize(k);
        for (int j_idx = 0; j_idx < k; ++j_idx)
        {
            donor_indices[j_idx] = distance_index_pairs[j_idx].second;
        }


    }
    else if (mixing_pool_mode == 15 || mixing_pool_mode == 16)
    {
        donor_indices = predeterminedMixingPools[solution_index];
    }
    else
    {
        std::cerr << "Unknown mixing pool mode " << mixing_pool_mode << " with knn mode " << mixing_pool_knn_k_mode << "." << std::endl;
        exit(1);
    }
}

bool FImodeRequiresDirection()
{
    if (fi_mode >= 1 ) return true;
    return false;
}

std::tuple<int, char**, double**, double*, void**> getPopulationKindAttributes(int kind)
{
    // Elitist archive
    if (kind == -1)
    {
        return std::tie(elitist_archive_size, elitist_archive, elitist_archive_objective_values, elitist_archive_constraint_values, elitist_archive_solution_metadata);
    }
    // Normal population
    if (kind == 0)
    {
        return std::tie(population_size, population, objective_values, constraint_values, solution_metadata);
    }

    std::cerr << "Unknown population kind " << kind << "." << std::endl;
    std::exit(1);
}

std::vector<int> getNearerBetterSolutions(int solution_idx, double* u)
{
    // If u is zero, select a random direction
    bool is_u_all_zeroes = true;
    for (int i = 0; i < number_of_objectives && is_u_all_zeroes; ++i)
        is_u_all_zeroes = u[i] == 0.0;
    if (is_u_all_zeroes)
        for (int i = 0; i < number_of_objectives; ++i)
            u[i] = randomRealUniform01();

    double self_s = computeLinearScalarization(nadir_point.data(), u, objective_values[solution_idx], number_of_objectives, objective_ranges);
    std::vector<int> indices(population_size);
    setVectorToRandomOrdering(indices);
    // std::iota(indices.begin(), indices.end(), 0);
    std::vector<double> scalarization(population_size);
    for (int j: indices)
        scalarization[j] = computeLinearScalarization(nadir_point.data(), u, objective_values[j], number_of_objectives, objective_ranges);
    
    // First, determine which solutions are better.
    auto new_end_indices = std::remove_if(indices.begin(), indices.end(), 
        [&self_s, &scalarization](int j)
        {
            double d = scalarization[j];
            return d <= self_s;
        });
    indices.resize(std::distance(indices.begin(), new_end_indices));
    if (indices.size() > 0)
    {
        // All elements where the scalarization is worse should have been removed.
        if (!(scalarization[indices[0]] >= self_s))
        {
            std::cerr << "uh oh item at index 0 (" << indices[0] << ") has scalarization " << scalarization[indices[0]] << " whereas current scalarizes as " << self_s << " which should have been worse." << std::endl;
            std::cerr << "nadir point is ";
            for (int i = 0; i < number_of_objectives; ++i)
            {
                std::cerr << nadir_point[i] << ", ";
            }
            std::cerr << "u is ";
            for (int i = 0; i < number_of_objectives; ++i)
            {
                std::cerr << u[i] << ", ";
            }
            std::cerr << "objective_ranges is ";
            for (int i = 0; i < number_of_objectives; ++i)
            {
                std::cerr << objective_ranges[i] << ", ";
            }
        }
        assert(scalarization[indices[0]] >= self_s);
    }
    // Compute relevant distances.
    std::vector<int> distances(population_size);
    for (int j: indices)
        distances[j] = hammingDistanceInParameterSpace(population[solution_idx], population[j]);
    // Sort by distances
    auto hamming = [&distances](int i, int j)
    {
        return distances[i] < distances[j];
    };
    std::sort(indices.begin(), indices.end(), hamming);
    if (indices.size() > 1)
        assert(distances[indices[0]] <= distances[indices[1]]);
    // , 
    // Prune: preserve only the elements that are undominated in (scalarization, distance)
    // Where higher is better for the first one, and lower is better for the second one.
    int current_scalarization = scalarization[indices[0]];
    new_end_indices = std::remove_if(indices.begin(), indices.end(), [&current_scalarization, &scalarization](int j){
        bool r = current_scalarization > scalarization[j];
        if (!r)
            current_scalarization = scalarization[j];
        return r;
    });
    indices.resize(std::distance(indices.begin(), new_end_indices));
    return indices;
}

std::tuple<int, int> getForcedImprovementDonor(int solution_idx, int cluster_idx, int objective_index, double* u, int l_fi_mode=fi_mode)
{
    if (l_fi_mode < 0)
    {
        std::cerr << "Elitist requested while Forced Improvement is disabled." << std::endl;
        std::exit(1);
    }

    // For single objective runs, always use single objective selection.
    if ((fi_use_singleobjective & 1) > 0 && objective_index >= 0)
    {
        int donor_index = 0;
        for (int j = 0; j < elitist_archive_size; j++)
        {
            if(optimization[objective_index] == MINIMIZATION)
            {
                if(elitist_archive_objective_values[j][objective_index] < elitist_archive_objective_values[donor_index][objective_index])
                    donor_index = j;
            }
            else if(optimization[objective_index] == MAXIMIZATION)
            {
                if(elitist_archive_objective_values[j][objective_index] > elitist_archive_objective_values[donor_index][objective_index])
                    donor_index = j;   
            }
        }
        return std::make_tuple(donor_index, -1);
    }

    // Otherwise use FI Mode.
    if (l_fi_mode == 0)
    {
        return std::make_tuple(randomInt(elitist_archive_size), -1);
    }
    else if (l_fi_mode == 1 || l_fi_mode == 2)
    {
        assert(u != NULL);
        int donor_index = 0;
        double min_d = INFINITY;
        for (int j = 0; j < elitist_archive_size; j++)
        {
            double d = euclideanDistanceToRay(nadir_point.data(), u, elitist_archive_objective_values[j], number_of_objectives, objective_ranges);
            if (d < min_d)
            {
                min_d = d;
                donor_index = j;
            }
        }
        return std::make_tuple(donor_index, -1);
    }
    else if (l_fi_mode == 10 || l_fi_mode == 11 || l_fi_mode == 12 || l_fi_mode == 13)
    {
        int line_idx = -1;
        std::vector<int> donor_indices;
        int old_mixing_pool_knn_k_mode = mixing_pool_knn_k_mode;
        mixing_pool_knn_k_mode = -3; 
        determineMixingPoolForSolution(solution_idx, cluster_idx, line_idx, donor_indices);
        mixing_pool_knn_k_mode = old_mixing_pool_knn_k_mode;

        int donor_index = 0;
        double max_d = -INFINITY;
        for (int j_idx = 0; j_idx < static_cast<int>(donor_indices.size()); j_idx++)
        {
            int j = donor_indices[j_idx];

            double d = computeLinearScalarization(nadir_point.data(), u, objective_values[j], number_of_objectives, objective_ranges);
            if (d > max_d)
            {
                max_d = d;
                donor_index = j;
            }
        }

        #ifdef DEBUG_PRINT
        static int last_solution_idx = -1;
        static int last_donor_idx = -1;

        if (last_solution_idx != solution_idx && last_donor_idx != donor_index)
        {
            double current_d = computeLinearScalarization(nadir_point.data(), u, objective_values[solution_idx], number_of_objectives, objective_ranges);
            std::cout << "solution " << solution_idx << " (";
            for (int o = 0; o < number_of_objectives; ++o)
            {
                std::cout << objective_values[solution_idx][o];
                if (o != number_of_objectives - 1)
                    std::cout << ", ";
                else
                    std::cout << ") ";
            }
            std::cout << "with scalarization " << current_d;
            std::cout << " will undergo FI with solution " << donor_index << " (";
            for (int o = 0; o < number_of_objectives; ++o)
            {
                std::cout << objective_values[donor_index][o];
                if (o != number_of_objectives - 1)
                    std::cout << ", ";
                else
                    std::cout << ") ";
            }
            std::cout << " and scalarization " << max_d << ".\n";

            last_solution_idx = solution_idx;
            last_donor_idx = donor_index;
        }
        #endif
        return std::make_tuple(donor_index, 0);
    }
    else if (l_fi_mode == 14 || l_fi_mode == 15)
    {
        int selector = 0;
        int fallback = 0;
        if (fi_mode == 15) selector = 1;

        std::vector<int> indices = getNearerBetterSolutions(solution_idx, u);
        
        // Nearest-Better like FI.
        if (indices.size() == 0)
            return getForcedImprovementDonor(solution_idx, cluster_idx, objective_index, u, fallback);

        if (selector == 0)
            return std::make_tuple(indices[0], 0);
        else if (selector == 1)
            return std::make_tuple(indices[randomInt(indices.size())], 0);
    }
    
    std::cerr << "Elitist requested for unknown FI mode." << std::endl;
    std::exit(1);
}

std::tuple<int, int> getForcedImprovementReplacement(int solution_idx, int cluster_idx, int objective_index, double* u, int l_fi_mode=fi_mode)
{
    if (l_fi_mode < 0)
    {
        std::cerr << "Elitist requested while Forced Improvement is disabled." << std::endl;
        std::exit(1);
    }

    // For single objective runs, always use single objective selection.
    if ((fi_use_singleobjective & 2) > 0 && objective_index >= 0)
    {
        int donor_index = 0;
        for (int j = 0; j < elitist_archive_size; j++)
        {
            if(optimization[objective_index] == MINIMIZATION)
            {
                if(elitist_archive_objective_values[j][objective_index] < elitist_archive_objective_values[donor_index][objective_index])
                    donor_index = j;
            }
            else if(optimization[objective_index] == MAXIMIZATION)
            {
                if(elitist_archive_objective_values[j][objective_index] > elitist_archive_objective_values[donor_index][objective_index])
                    donor_index = j;   
            }
        }
        return std::make_tuple(donor_index, -1);
    }

    if (l_fi_mode == 0 || l_fi_mode == 1 || l_fi_mode == 11 )
    {
        return std::make_tuple(randomInt(elitist_archive_size), -1);
    }
    else if (l_fi_mode == 12 )
    {
        return std::make_tuple(randomInt(population_size), 0);
    }
    else if (l_fi_mode == 13)
    {
        // Attempt to preserve diversity by selecting the farthest one
        // Out of a 3-tournament with one
        int a = randomInt(population_size);
        int b = randomInt(population_size);
        int c = randomInt(population_size);
        int h_ab = hammingDistanceInParameterSpace(population[a], population[b]);
        int h_ac = hammingDistanceInParameterSpace(population[a], population[c]);
        int h_bc = hammingDistanceInParameterSpace(population[b], population[c]);
        int hs_a = h_ab + h_ac;
        int hs_b = h_ab + h_bc;
        int hs_c = h_ac + h_bc;
        if (hs_a >= hs_b)
            if (hs_c >= hs_a)
                return std::make_tuple(c, 0);
            else
                return std::make_tuple(a, 0);
        else
            if (hs_c >= hs_b)
                return std::make_tuple(c, 0);
            else
                return std::make_tuple(b, 0);
    }
    else if (l_fi_mode == 14 || l_fi_mode == 15)
    {
        // From the population find the sample as close as possible
        // to the sample from the archive.
        int a = randomInt(elitist_archive_size);
        // We are going to this in a multi-objective fashion
        // First list all indices.
        std::vector<int> indices(population_size);
        // Sort by euclidean distance to reference in objective space
        std::vector<double> euclidean(population_size);
        for (int i = 0; i < population_size; ++i)
            euclidean[i] = distanceEuclidean(objective_values[i], elitist_archive_objective_values[a], number_of_objectives, objective_ranges);
        std::sort(indices.begin(), indices.end(), 
        [&euclidean](int x, int y)
        {
            return euclidean[x] < euclidean[y];
        });
        // We want to minimize the hamming distance to the original solution
        // (to preserve the chance that we'll stay within the same niche)
        int current_hamming = number_of_parameters;
        auto new_end = std::remove_if(indices.begin(), indices.end(),
        [&solution_idx, &current_hamming](int x)
        {
            int h_x = hammingDistanceInParameterSpace(population[x], population[solution_idx]);
            if (h_x < current_hamming)
            {
                current_hamming = h_x;
                return false;
            }
            return true;
        });
        indices.erase(new_end, indices.end());

        // Add marker for sampling from archive instead.
        indices.push_back(-1);
        int idx = indices[randomInt(indices.size())];
        if (idx >= 0)
            return std::make_tuple(idx, 0);
        else
            return std::make_tuple(a, -1);
    }
    else if (false)
    {
        int selector = 0;
        int fallback = 0;
        if (l_fi_mode == 15) selector = 1;

        std::vector<int> indices = getNearerBetterSolutions(solution_idx, u);
        
        // Nearest-Better like FI.
        if (indices.size() == 0)
            return getForcedImprovementReplacement(solution_idx, cluster_idx, objective_index, u, fallback);

        if (selector == 0)
            return std::make_tuple(indices[0], 0);
        else if (selector == 1)
            return std::make_tuple(indices[randomInt(indices.size())], 0);
    }
    else if (l_fi_mode == 2)
    {
        assert(u != NULL);
        int donor_index = 0;
        double min_d = INFINITY;
        for (int j = 0; j < elitist_archive_size; j++)
        {
            double d = euclideanDistanceToRay(nadir_point.data(), u, elitist_archive_objective_values[j], number_of_objectives, objective_ranges);
            if (d < min_d)
            {
                min_d = d;
                donor_index = j;
            }
        }
        return std::make_tuple(donor_index, -1);
    }
    else if (l_fi_mode == 10)
    {
        int line_idx = -1;
        std::vector<int> donor_indices;
        determineMixingPoolForSolution(solution_idx, cluster_idx, line_idx, donor_indices);
        
        int donor_index = 0;
        double min_d = INFINITY;
        for (int j_idx = 0; j_idx < static_cast<int>(donor_indices.size()); j_idx++)
        {
            int j = donor_indices[j_idx];

            double d = computeLinearScalarization(nadir_point.data(), u, objective_values[j], number_of_objectives, objective_ranges);
            if (d < min_d)
            {
                min_d = d;
                donor_index = j;
            }
        }

        return std::make_tuple(donor_index, 0);
    }
    
    std::cerr << "Elitist requested for unknown FI mode." << std::endl;
    std::exit(1);
}

/**
 * Multi-objective Line Gene-pool Optimal Mixing
 * Construct an offspring from a parent solution using linear line scalarization.
 */
void performLineMultiObjectiveGenepoolOptimalMixing( int cluster_idx, int line_index, int solution_idx, char *parent, double *parent_obj, double parent_con, void* parent_metadata, 
                            char *result, double *obj, double *con, void *metadata)
{
    char   *backup, *donor, is_unchanged, changed, improved, is_improved, is_strictly_improved, replaced, is_new_nondominated_point, is_dominated_by_archive;
    int     i, j, donor_index, *order, linkage_group_index, number_of_linkage_sets;
    double  *obj_backup, con_backup;

    void *metadata_backup = NULL;
    initSolutionMetadata(metadata_backup);

    int line_idx_x = line_index;
    if (use_cluster_line_instead_of_assigned_line_for_line_mixing)
        line_idx_x = cluster_idx;

    /* Clone the parent solution. */
    copyFromAToB(parent, parent_obj, parent_con, parent_metadata, result, obj, con, metadata);
    // Offspring is now initialized: update variable.
    initialized_offspring_size = solution_idx;
    
    /* Create a backup version of the parent solution. */
    backup = (char *) Malloc( number_of_parameters*sizeof( char ) );
    obj_backup = (double *) Malloc( number_of_objectives*sizeof( double ) );
    copyFromAToB(result, obj, *con, metadata, backup, obj_backup, &con_backup, metadata_backup);

    number_of_linkage_sets = lt_length[cluster_idx]; // Root is already removed from the tree - 1; /* Remove root from the linkage tree. */
    order = createRandomOrdering(number_of_linkage_sets);
    
    /* Determine the pool of indices in the population with which we can mix */
    std::vector<int> donor_indices;
    determineMixingPoolForSolution(solution_idx, cluster_idx, line_index, donor_indices);

    if(donor_indices.size() == 0)
    {
        // Population has converged, exit early
        return;
    }

    /* Traverse the linkage tree for Gene-pool Optimal Mixing */
    changed = FALSE;
    improved = FALSE;
    replaced = FALSE;
    for( i = 0; i < number_of_linkage_sets; i++ )
    {
        linkage_group_index = order[i];

        // Grab a random solution from the donor pool.
        is_unchanged = TRUE;
        for (int donor_indices_left = donor_indices.size(); donor_indices_left > 0; --donor_indices_left)
        {
            // Grab a random solution from the donor pool.
            int donor_indices_index = randomInt(donor_indices_left);
            donor_index = donor_indices[donor_indices_index];
            std::swap(donor_indices[donor_indices_index], donor_indices[donor_indices_left - 1]);
            donor = population[donor_index];

            copyValuesFromDonorToOffspring(result, donor, cluster_idx, linkage_group_index);     
            mutateSolution(result, linkage_group_index, cluster_idx);

            /* Check if the new intermediate solution is different from the previous state. */
            is_unchanged = TRUE;
            for( j = 0; j < lt_number_of_indices[cluster_idx][linkage_group_index]; j++ )
            {
                if( backup[lt[cluster_idx][linkage_group_index][j]] != result[lt[cluster_idx][linkage_group_index][j]] )
                {
                    is_unchanged = FALSE;
                    break;
                }
            }

            // Break if solution was changed.
            // Or donor search is not enabled
            if ( is_unchanged == FALSE || !use_donor_search ) break;
        }

        if( is_unchanged == FALSE )
        {
            // Determine line.
            double* line_origin;
            double* line_direction;
            std::vector<double> u;
            determineLineDirection(line_idx_x, obj_backup, obj, line_origin, line_direction, u);

            is_improved = FALSE;
            is_strictly_improved = FALSE;          
            evaluateIndividual(result, obj, con, metadata, NOT_EXTREME_CLUSTER);
            updateElitistArchiveWithReplacementOfExistedMember(result, obj, *con, metadata, &is_new_nondominated_point, &is_dominated_by_archive);

            double perf_new = projectedLineObjectiveDistance(line_origin, line_direction, obj);
            double perf_old = projectedLineObjectiveDistance(line_origin, line_direction, obj_backup);
            
            // DEBUG:
            // if (donor_index == 0 && perf_new > perf_old)
            // {
            //     std::cout << "Is (";
            //     for (int d = 0; d < number_of_objectives; ++d)
            //     {
            //         if (d != 0) std::cout << ", ";
            //         std::cout << obj[d];
            //     }
            //     std::cout << ") " << perf_new << " better than (";
            //     for (int d = 0; d < number_of_objectives; ++d)
            //     {
            //         if (d != 0) std::cout << ", ";
            //         std::cout << obj_backup[d];
            //     }
            //     std::cout << ") " << perf_old << "?" << std::endl;
            // }

            // If infeasible: Lower constraint is better.
            if ((* con) > 0 && con_backup > 0 && (* con) < con_backup)
                is_strictly_improved = TRUE;
            // Otherwise: use scalarization to compare fitness.
            else if (perf_new > perf_old)
                is_strictly_improved = TRUE;
            // MAYBE: Allow equal fitness change.
            else if (perf_new >= perf_old)
                is_improved = TRUE;

            /* Check for weak Pareto domination. */
            if ( constraintWeaklyParetoDominates( obj, *con, obj_backup, con_backup) )
                is_improved = TRUE;
            /* Strict improvement! */
            if ( constraintParetoDominates( obj, *con, obj_backup, con_backup) )
                is_strictly_improved = TRUE;

            /* Check if the new intermediate solution is dominated by any solution in the archive. 
                Note that if the new intermediate solution is a point in the archive, it is NOT considered to be dominated by the archive.*/
            if ( !is_dominated_by_archive )
                is_strictly_improved = TRUE;
            
            /* If a solution is strictly improved, it is certainly improved as well. */
            if (is_strictly_improved)
                is_improved = TRUE;

            if ( is_improved )
            {
                changed = TRUE;
                copyFromAToB(result, obj, *con, metadata, backup, obj_backup, &con_backup, metadata_backup);
            }
            else
                copyFromAToB(backup, obj_backup, con_backup, metadata_backup, result, obj, con, metadata);

            // This improved means a positive change in objective value, which is an
            // as an irreversible change.
            if (is_strictly_improved)
                improved = TRUE;
        }
    }
    free(order);

    /* Forced Improvement */
    if ((fi_mode >= 0) && ((!changed) || (use_solution_NIS_instead_of_population_NIS
                                              ? (s_NIS[solution_idx] > (1 + floor(log10(population_size))))
                                              : (t_NIS > (1 + floor(log10(population_size)))))))
    {
        // Note, direction vector for the original objective is always the same for FI.
        // No need to recompute every iteration / every subset.
        std::vector<double> u_fi;
        if (FImodeRequiresDirection())
            computeNormalizedDirectionVector(u_fi, nadir_point.data(), obj, objective_ranges);
        
        changed = FALSE;
        order = createRandomOrdering(number_of_linkage_sets);
        /* Perform another round of Gene-pool Optimal Mixing with the donors randomly selected from the archive. */
        for(i = 0; i < number_of_linkage_sets; i++)
        {
            int population_kind;
            std::tie(donor_index, population_kind) = getForcedImprovementDonor(solution_idx, cluster_idx, -1, u_fi.data());

            char **ip_population;
            std::tie(std::ignore, ip_population, std::ignore, std::ignore,
                     std::ignore) = getPopulationKindAttributes(population_kind);

            linkage_group_index = order[i];
            copyValuesFromDonorToOffspring(result, ip_population[donor_index], cluster_idx, linkage_group_index);
            mutateSolution(result, linkage_group_index, cluster_idx);

            /* Check if the new intermediate solution is different from the previous state. */
            is_unchanged = TRUE;    
            for( j = 0; j < lt_number_of_indices[cluster_idx][linkage_group_index]; j++ )
            {
                if( backup[lt[cluster_idx][linkage_group_index][j]] != result[lt[cluster_idx][linkage_group_index][j]] )
                {
                    is_unchanged = FALSE;
                    break;
                }
            }           

            if( is_unchanged == FALSE )
            {      
                // Determine line.
                double* line_origin;
                double* line_direction;
                std::vector<double> u;
                determineLineDirection(line_idx_x, obj_backup, obj, line_origin, line_direction, u);

                double perf_new = projectedLineObjectiveDistance(line_origin, line_direction, obj);
                double perf_old = projectedLineObjectiveDistance(line_origin, line_direction, obj_backup);
 
                is_improved = FALSE;

                // If infeasible: Lower constraint is better.
                if ((* con) > 0 && con_backup > 0 && (* con) < con_backup)
                    is_improved = TRUE;
                // Otherwise: use scalarization to compare fitness (strictly).
                else if (perf_new > perf_old)
                    is_improved = TRUE;

                /* Check for (strict) Pareto domination. */
                if ( constraintParetoDominates( obj, *con, obj_backup, con_backup) )
                    is_improved = TRUE;

                /* Check if a truly new non-dominated solution is created. */
                if(is_new_nondominated_point)
                    is_improved = TRUE;
                
                if ( is_improved )
                {
                    changed = TRUE;
                    // FI is always strict, hence improvement here always indicates a positive
                    // change in objective value.
                    improved = TRUE;
                    copyFromAToB(result, obj, *con, metadata, backup, obj_backup, &con_backup, metadata_backup);
                    break;
                }
                else
                    copyFromAToB(backup, obj_backup, con_backup, metadata_backup, result, obj, con, metadata);
            }
        }
        free(order);

        if(!changed && fi_do_replacement)
        {
            int population_kind;
            std::tie(donor_index, population_kind) = getForcedImprovementReplacement(solution_idx, cluster_idx, -1, u_fi.data());

            char **ip_population;
            double **ip_objective_values, *ip_constraint_values;
            void **ip_solution_metadata;
            std::tie(std::ignore, ip_population, ip_objective_values, ip_constraint_values,
                     ip_solution_metadata) = getPopulationKindAttributes(population_kind);

            copyFromAToB(ip_population[donor_index], ip_objective_values[donor_index], 
                    ip_constraint_values[donor_index], ip_solution_metadata[donor_index],
                    result, obj, con, metadata);

            replaced = TRUE;
        }
    }

    
    if (
        (((reset_nis_on & 1) > 0) && changed) ||
        (((reset_nis_on & 2) > 0) && improved) ||
        (((reset_nis_on & 4) > 0) && replaced)
    )
        s_NIS[solution_idx] = 0;
    else
        s_NIS[solution_idx] += 1;

    free( backup ); free( obj_backup );
    cleanupSolutionMetadata(metadata_backup);
}


/**
 * Multi-objective Gene-pool Optimal Mixing
 * Construct an offspring from a parent solution in a middle-region cluster.
 */
void performMultiObjectiveGenepoolOptimalMixing( int cluster_idx, int solution_idx, char *parent, double *parent_obj, double parent_con, void* parent_metadata,
                            char *result, double *obj,  double *con, void* metadata)
{
    char   *backup, *donor, is_unchanged, changed, improved, is_improved, is_strictly_improved, replaced, is_new_nondominated_point, is_dominated_by_archive;
    int     i, j, donor_index, *order, linkage_group_index, number_of_linkage_sets;
    double  *obj_backup, con_backup;

    void *metadata_backup = NULL;
    initSolutionMetadata(metadata_backup);

    /* Clone the parent solution. */
    copyFromAToB(parent, parent_obj, parent_con, parent_metadata, result, obj, con, metadata);
    // Offspring is now initialized: update variable.
    initialized_offspring_size = solution_idx;

    /* Create a backup version of the parent solution. */
    backup = (char *) Malloc( number_of_parameters*sizeof( char ) );
    obj_backup = (double *) Malloc( number_of_objectives*sizeof( double ) );
    copyFromAToB(result, obj, *con, metadata, backup, obj_backup, &con_backup, metadata_backup);

    number_of_linkage_sets = lt_length[cluster_idx]; // Root is already removed from the tree - 1; /* Remove root from the linkage tree. */
    order = createRandomOrdering(number_of_linkage_sets);

    /* Determine the pool of indices in the population with which we can mix */
    std::vector<int> donor_indices;
    determineMixingPoolForSolution(solution_idx, cluster_idx, -1, donor_indices);

    if(donor_indices.size() == 0)
    {
        // Population has converged, exit early
        return;
    }
    
    /* Traverse the linkage tree for Gene-pool Optimal Mixing */
    changed = FALSE;
    improved = FALSE;
    replaced = FALSE;
    for( i = 0; i < number_of_linkage_sets; i++ )
    {
        linkage_group_index = order[i];

        is_unchanged = TRUE;
        for (int donor_indices_left = donor_indices.size(); donor_indices_left > 0; --donor_indices_left)
        {
            // Grab a random solution from the donor pool.
            int donor_indices_index = randomInt(donor_indices_left);
            donor_index = donor_indices[donor_indices_index];
            std::swap(donor_indices[donor_indices_index], donor_indices[donor_indices_left - 1]);
            donor = population[donor_index];

            copyValuesFromDonorToOffspring(result, donor, cluster_idx, linkage_group_index);     
            mutateSolution(result, linkage_group_index, cluster_idx);

            /* Check if the new intermediate solution is different from the previous state. */
            is_unchanged = TRUE;
            for( j = 0; j < lt_number_of_indices[cluster_idx][linkage_group_index]; j++ )
            {
                if( backup[lt[cluster_idx][linkage_group_index][j]] != result[lt[cluster_idx][linkage_group_index][j]] )
                {
                    is_unchanged = FALSE;
                    break;
                }
            }
        
            // Break if solution was changed.
            if ( is_unchanged == FALSE || !use_donor_search ) break;
        }

        if( is_unchanged == FALSE )
        {
            is_improved = FALSE;
            is_strictly_improved = FALSE;
            evaluateIndividual(result, obj, con, metadata, NOT_EXTREME_CLUSTER);
            updateElitistArchiveWithReplacementOfExistedMember(result, obj, *con, metadata, &is_new_nondominated_point, &is_dominated_by_archive);

            /* Check for weak Pareto domination. */
            if ( constraintWeaklyParetoDominates( obj, *con, obj_backup, con_backup) )
                is_improved = TRUE;
            if ( constraintParetoDominates( obj, *con, obj_backup, con_backup) )
                is_strictly_improved = TRUE;

            /* Check if the new intermediate solution is dominated by any solution in the archive. 
                Note that if the new intermediate solution is a point in the archive, it is NOT considered to be dominated by the archive.*/
            if ( !is_dominated_by_archive )
                is_strictly_improved = TRUE;
            
            if ( is_strictly_improved )
                is_improved = TRUE;

            if ( is_improved )
            {
                changed = TRUE;
                copyFromAToB(result, obj, *con, metadata,  backup, obj_backup, &con_backup, metadata_backup);
            }
            else
                copyFromAToB(backup, obj_backup, con_backup, metadata_backup, result, obj, con, metadata);
        
            // This improved means a positive change in objective value, which is an
            // as an irreversible change.
            if ( is_strictly_improved )
                improved = TRUE;
        }
    }
    free(order);

    /* Forced Improvement */
    if ((fi_mode >= 0) && ((!changed) || (use_solution_NIS_instead_of_population_NIS
                                              ? (s_NIS[solution_idx] > (1 + floor(log10(population_size))))
                                              : (t_NIS > (1 + floor(log10(population_size)))))))
    {
        // Note, direction vector for the original objective is always the same for FI.
        // No need to recompute every iteration / every subset.
        std::vector<double> u_fi;
        if (FImodeRequiresDirection())
            computeNormalizedDirectionVector(u_fi, nadir_point.data(), obj, objective_ranges);

        changed = FALSE;
        order = createRandomOrdering(number_of_linkage_sets);
        /* Perform another round of Gene-pool Optimal Mixing with the donors randomly selected from the archive. */
        for(i = 0; i < number_of_linkage_sets; i++)
        {
            int population_kind;
            std::tie(donor_index, population_kind) = getForcedImprovementDonor(solution_idx, cluster_idx, -1, u_fi.data());

            char **ip_population;
            std::tie(std::ignore, ip_population, std::ignore, std::ignore,
                     std::ignore) = getPopulationKindAttributes(population_kind);

            linkage_group_index = order[i];
            copyValuesFromDonorToOffspring(result, ip_population[donor_index], cluster_idx, linkage_group_index);
            mutateSolution(result, linkage_group_index, cluster_idx);

            /* Check if the new intermediate solution is different from the previous state. */
            is_unchanged = TRUE;    
            for( j = 0; j < lt_number_of_indices[cluster_idx][linkage_group_index]; j++ )
            {
                if( backup[lt[cluster_idx][linkage_group_index][j]] != result[lt[cluster_idx][linkage_group_index][j]] )
                {
                    is_unchanged = FALSE;
                    break;
                }
            }           

            if( is_unchanged == FALSE )
            {
                is_improved = FALSE;

                evaluateIndividual(result, obj, con, metadata, NOT_EXTREME_CLUSTER);
                updateElitistArchiveWithReplacementOfExistedMember(result, obj, *con, metadata, &is_new_nondominated_point, &is_dominated_by_archive);

                /* Check for (strict) Pareto domination. */
                if ( constraintParetoDominates( obj, *con, obj_backup, con_backup) )
                    is_improved = TRUE;

                /* Check if a truly new non-dominated solution is created. */
                if(is_new_nondominated_point)
                    is_improved = TRUE;
                
                if ( is_improved )
                {
                    changed = TRUE;
                    improved = TRUE;
                    copyFromAToB(result, obj, *con, metadata,  backup, obj_backup, &con_backup, metadata_backup);
                    break;
                }
                else
                    copyFromAToB(backup, obj_backup, con_backup, metadata_backup, result, obj, con, metadata);
            }
        }
        free(order);

        if(!changed && fi_do_replacement)
        {
            int population_kind;
            std::tie(donor_index, population_kind) = getForcedImprovementReplacement(solution_idx, cluster_idx, -1, u_fi.data());

            char **ip_population;
            double **ip_objective_values, *ip_constraint_values;
            void **ip_solution_metadata;
            std::tie(std::ignore, ip_population, ip_objective_values, ip_constraint_values,
                     ip_solution_metadata) = getPopulationKindAttributes(population_kind);

            copyFromAToB(ip_population[donor_index], ip_objective_values[donor_index], 
                    ip_constraint_values[donor_index], ip_solution_metadata[donor_index],
                    result, obj, con, metadata);

            replaced = TRUE;
        }

        if (
            (((reset_nis_on & 1) > 0) && changed) ||
            (((reset_nis_on & 2) > 0) && improved) ||
            (((reset_nis_on & 4) > 0) && replaced)
        )
            s_NIS[solution_idx] = 0;
        else
            s_NIS[solution_idx] += 1;
    }

    free( backup ); free( obj_backup );
    cleanupSolutionMetadata(metadata_backup);
}
/**
 * Single-objective Gene-pool Optimal Mixing
 * Construct an offspring from a parent solution in an extreme-region cluster.
 */
void performSingleObjectiveGenepoolOptimalMixing( int cluster_idx, int objective_index, int solution_idx,
                                char *parent, double *parent_obj, double parent_con, void* parent_metadata,
                                char *result, double *obj, double *con, void* metadata )
{
    char   *backup, *donor, *elitist_copy, is_unchanged, changed, improved, is_improved, is_strictly_improved, replaced, is_new_nondominated_point, is_dominated_by_archive;
    int     i, j, donor_index, number_of_linkage_sets, linkage_group_index, *order;
    double  *obj_backup, con_backup;

    void *metadata_backup = NULL;
    initSolutionMetadata(metadata_backup);

    /* Clone the parent solution. */
    copyFromAToB(parent, parent_obj, parent_con, parent_metadata, result, obj, con, metadata);
    // Offspring is now initialized: update variable.
    initialized_offspring_size = solution_idx;

    /* Create a backup version of the parent solution. */
    backup = (char *) Malloc( number_of_parameters*sizeof( char ) );
    obj_backup = (double *) Malloc( number_of_objectives*sizeof( double ) );
    copyFromAToB(result, obj, *con, metadata, backup, obj_backup, &con_backup, metadata_backup);

    number_of_linkage_sets = lt_length[cluster_idx]; // Root is already removed from the tree - 1; /* Remove root from the linkage tree. */
    
    order = createRandomOrdering(number_of_linkage_sets);

    /* Determine the pool of indices in the population with which we can mix */
    std::vector<int> donor_indices;
    determineMixingPoolForSolution(solution_idx, cluster_idx, -1, donor_indices);

    if(donor_indices.size() == 0)
    {
        // Population has converged, exit early
        return;
    }

    /* Traverse the linkage tree for Gene-pool Optimal Mixing */
    changed = FALSE;
    improved = FALSE;
    replaced = FALSE;
    for( i = 0; i < number_of_linkage_sets; i++ )
    {
        linkage_group_index = order[i];
        
        is_unchanged = TRUE;
        for (int donor_indices_left = donor_indices.size(); donor_indices_left > 0; --donor_indices_left)
        {
            // Grab a random solution from the donor pool.
            int donor_indices_index = randomInt(donor_indices_left);
            donor_index = donor_indices[donor_indices_index];
            std::swap(donor_indices[donor_indices_index], donor_indices[donor_indices_left - 1]);
            donor = population[donor_index];

            copyValuesFromDonorToOffspring(result, donor, cluster_idx, linkage_group_index);        
            mutateSolution(result, linkage_group_index, cluster_idx);

            /* Check if the new intermediate solution is different from the previous state. */
            is_unchanged = TRUE;
            for( j = 0; j < lt_number_of_indices[cluster_idx][linkage_group_index]; j++ )
            {
                if( backup[lt[cluster_idx][linkage_group_index][j]] != result[lt[cluster_idx][linkage_group_index][j]] )
                {
                    is_unchanged = FALSE;
                    break;
                }
            }

            // Break if solution was changed.
            // Or donor search is not enabled
            if ( is_unchanged == FALSE || !use_donor_search ) break;
        }

        if( is_unchanged == FALSE )
        {
            is_improved = FALSE;
            is_strictly_improved = FALSE;
            evaluateIndividual(result, obj, con, metadata, objective_index);
            updateElitistArchiveWithReplacementOfExistedMember(result, obj, *con, metadata, &is_new_nondominated_point, &is_dominated_by_archive);

            if (betterFitness(obj, *con, obj_backup, con_backup, objective_index) )
                is_strictly_improved = TRUE;
            else if (equalFitness(obj, *con, obj_backup, con_backup, objective_index) )
                is_improved = TRUE;
            
            if ( is_strictly_improved )
                is_improved = TRUE;

            if ( is_improved )
            {
                changed = TRUE;
                copyFromAToB(result, obj, *con, metadata, backup, obj_backup, &con_backup, metadata_backup);
            }
            else
                copyFromAToB(backup, obj_backup, con_backup, metadata_backup, result, obj, con, metadata);    
        
            if ( is_strictly_improved )
                improved = TRUE;
        }
    }
    free(order);

    elitist_copy = (char*)Malloc(number_of_parameters*sizeof(char));
    /* Forced Improvement*/
    if ((fi_mode >= 0) && ((!changed) || ((use_solution_NIS_instead_of_population_NIS
                                               ? (s_NIS[solution_idx] > (1 + floor(log10(population_size))))
                                               : (t_NIS > (1 + floor(log10(population_size))))))))
    {
        std::vector<double> u_fi;
        if (FImodeRequiresDirection())
            computeNormalizedDirectionVector(u_fi, nadir_point.data(), obj, objective_ranges);

        changed = FALSE;
        
        /* Find in the archive the solution having the best value in the corresponding objective. */
        int population_kind;
        std::tie(donor_index, population_kind) = getForcedImprovementDonor(solution_idx, cluster_idx, objective_index, u_fi.data());

        char **ip_population;
        std::tie(std::ignore, ip_population, std::ignore, std::ignore,
                    std::ignore) = getPopulationKindAttributes(population_kind);

        for (j = 0; j < number_of_parameters; j++)
            elitist_copy[j] = ip_population[donor_index][j];

        /* Perform Gene-pool Optimal Mixing with the single-objective best-found solution as the donor. */
        order = createRandomOrdering(number_of_linkage_sets);
        for( i = 0; i < number_of_linkage_sets; i++ )
        {
            linkage_group_index = order[i];
            copyValuesFromDonorToOffspring(result, elitist_copy, cluster_idx, linkage_group_index);
            mutateSolution(result, linkage_group_index, cluster_idx);

            /* Check if the new intermediate solution is different from the previous state. */
            is_unchanged = TRUE;
            for( j = 0; j < lt_number_of_indices[cluster_idx][linkage_group_index]; j++ )
            {
                if( backup[lt[cluster_idx][linkage_group_index][j]] != result[lt[cluster_idx][linkage_group_index][j]] )
                {
                    is_unchanged = FALSE;
                    break;
                }
            }           

            if( is_unchanged == FALSE )
            {
                is_improved = FALSE;
                evaluateIndividual(result, obj, con, metadata, objective_index);
                updateElitistArchiveWithReplacementOfExistedMember(result, obj, *con, metadata, &is_new_nondominated_point, &is_dominated_by_archive);

                /* Check if strict improvement in the corresponding objective. */
                if(betterFitness(obj, *con, obj_backup, con_backup, objective_index) )
                    is_improved = TRUE;

                if (is_improved == TRUE)
                {
                    changed = TRUE;
                    improved = TRUE;
                    copyFromAToB(result, obj, *con, metadata, backup, obj_backup, &con_backup, metadata_backup );
                    break;
                }
                else
                    copyFromAToB(backup, obj_backup, con_backup, metadata_backup, result, obj, con, metadata);
            }
        }
        free(order);

        if(!changed && fi_do_replacement)
        {
            int population_kind;
            std::tie(donor_index, population_kind) = getForcedImprovementReplacement(solution_idx, cluster_idx, objective_index, NULL);

            char **ip_population;
            double **ip_objective_values, *ip_constraint_values;
            void **ip_solution_metadata;
            std::tie(std::ignore, ip_population, ip_objective_values, ip_constraint_values,
                     ip_solution_metadata) = getPopulationKindAttributes(population_kind);

            copyFromAToB(ip_population[donor_index], ip_objective_values[donor_index], 
                    ip_constraint_values[donor_index], ip_solution_metadata[donor_index],
                    result, obj, con, metadata);

            replaced = TRUE;
        }
    }

    if (
        (((reset_nis_on & 1) > 0) && changed) ||
        (((reset_nis_on & 2) > 0) && improved) ||
        (((reset_nis_on & 4) > 0) && replaced)
    )
        s_NIS[solution_idx] = 0;
    else
        s_NIS[solution_idx] += 1;
    
    free( backup ); free( obj_backup ); free( elitist_copy );
    cleanupSolutionMetadata(metadata_backup);
}

void updateIdealPointWithNewObjective(double *obj, double *reference)
{
    for (int d = 0; d < number_of_objectives; ++d)
    {
        if (optimization[d] == MAXIMIZATION)
        {
            reference[d] = std::max(reference[d], obj[d]);
        }
        else
        {
            reference[d] = std::min(reference[d], obj[d]);
        }
    }
}

/**
 * Multi-objective Tschebysheff Gene-pool Optimal Mixing
 * Construct an offspring from a parent solution in a middle-region cluster.
 */
void performTschebysheffObjectiveGenepoolOptimalMixing(int cluster_idx, int solution_idx, 
    char *parent, double *parent_obj, double parent_con, void* parent_metadata,
    char *result, double *obj, double *con, void* metadata, 
    double *tsch_weight_vector, double *reference, double *ranges)
{
    char   *backup, *donor, is_unchanged, changed, improved, is_improved, is_strictly_improved, replaced, is_new_nondominated_point, is_dominated_by_archive;
    int     i, j, donor_index, *order, linkage_group_index, number_of_linkage_sets;
    double  *obj_backup, con_backup;

    void *metadata_backup = NULL;
    initSolutionMetadata(metadata_backup);

    /* Clone the parent solution. */
    copyFromAToB(parent, parent_obj, parent_con, parent_metadata, result, obj, con, metadata);
    // Offspring is now initialized: update variable.
    initialized_offspring_size = solution_idx;

    /* Create a backup version of the parent solution. */
    backup = (char *) Malloc( number_of_parameters*sizeof( char ) );
    obj_backup = (double *) Malloc( number_of_objectives*sizeof( double ) );
    copyFromAToB(result, obj, *con, metadata, backup, obj_backup, &con_backup, metadata_backup);

    number_of_linkage_sets = lt_length[cluster_idx]; // Root is already removed from the tree - 1; /* Remove root from the linkage tree. */
    order = createRandomOrdering(number_of_linkage_sets);

    /* Determine the pool of indices in the population with which we can mix */
    std::vector<int> donor_indices;
    determineMixingPoolForSolution(solution_idx, cluster_idx, -1, donor_indices);

    if(donor_indices.size() == 0)
    {
        // Population has converged, exit early
        return;
    }
    
    /* Traverse the linkage tree for Gene-pool Optimal Mixing */
    changed = FALSE;
    improved = FALSE;
    replaced = FALSE;
    for( i = 0; i < number_of_linkage_sets; i++ )
    {
        linkage_group_index = order[i];

        is_unchanged = TRUE;
        for (int donor_indices_left = donor_indices.size(); donor_indices_left > 0; --donor_indices_left)
        {
            // Grab a random solution from the donor pool.
            int donor_indices_index = randomInt(donor_indices_left);
            donor_index = donor_indices[donor_indices_index];
            std::swap(donor_indices[donor_indices_index], donor_indices[donor_indices_left - 1]);
            donor = population[donor_index];

            copyValuesFromDonorToOffspring(result, donor, cluster_idx, linkage_group_index);     
            mutateSolution(result, linkage_group_index, cluster_idx);

            /* Check if the new intermediate solution is different from the previous state. */
            is_unchanged = TRUE;
            for( j = 0; j < lt_number_of_indices[cluster_idx][linkage_group_index]; j++ )
            {
                if( backup[lt[cluster_idx][linkage_group_index][j]] != result[lt[cluster_idx][linkage_group_index][j]] )
                {
                    is_unchanged = FALSE;
                    break;
                }
            }

            // Break if solution was changed.
            // Or donor search is not enabled
            if ( is_unchanged == FALSE || !use_donor_search ) break;
        }

        if( is_unchanged == FALSE )
        {
            is_improved = FALSE;
            is_strictly_improved = FALSE;
            evaluateIndividual(result, obj, con, metadata, NOT_EXTREME_CLUSTER);
            updateElitistArchiveWithReplacementOfExistedMember(result, obj, *con, metadata, &is_new_nondominated_point, &is_dominated_by_archive);

            /* Check for weak Pareto domination. */
            if ( constraintWeaklyParetoDominates( obj, *con, obj_backup, con_backup) )
                is_improved = TRUE;
            if ( constraintParetoDominates( obj, *con, obj_backup, con_backup) )
                is_strictly_improved = TRUE;

            /* Check if the new intermediate solution is dominated by any solution in the archive. 
                Note that if the new intermediate solution is a point in the archive, it is NOT considered to be dominated by the archive.*/
            if ( !is_dominated_by_archive )
                is_strictly_improved = TRUE;

            // Compute current scalarized Tschebysheff distance.
            // Update ideal point.
            updateIdealPointWithNewObjective(obj, reference);

            // Compute scalarized value (with the new )
            double tsch = computeTchebysheffDistance(number_of_objectives, obj, reference, tsch_weight_vector, ranges);
            double tsch_backup = computeTchebysheffDistance(number_of_objectives, obj_backup, reference, tsch_weight_vector, ranges);

            if ( tsch < tsch_backup )
                is_strictly_improved = true;
            // MAYBE: Allow equal change.
            // Note that for Tchebysheff scalarizations this may lead to a regression,
            // whereas all non-regressions are covered the the pareto-dominates condition.
            // So allowing this is likely not in our best interest.
            // else if (tsch <= tsch_backup)
            //    is_improved = true;

            if (is_strictly_improved)
                is_improved = true;

            if ( is_improved )
            {
                changed = TRUE;
                copyFromAToB(result, obj, *con, metadata,  backup, obj_backup, &con_backup, metadata_backup);
            }
            else
                copyFromAToB(backup, obj_backup, con_backup, metadata_backup, result, obj, con, metadata);
        
            if (is_strictly_improved)
                improved = TRUE;
        }
    }
    free(order);

    /* Forced Improvement */
    if ((fi_mode >= 0) && ((!changed) || (use_solution_NIS_instead_of_population_NIS
                                              ? (s_NIS[solution_idx] > (1 + floor(log10(population_size))))
                                              : (t_NIS > (1 + floor(log10(population_size)))))))
    {
        // Note, direction vector for the original objective is always the same for FI.
        // No need to recompute every iteration / every subset.
        std::vector<double> u_fi;
        if (FImodeRequiresDirection())
            computeNormalizedDirectionVector(u_fi, nadir_point.data(), obj, objective_ranges);

        changed = FALSE;
        order = createRandomOrdering(number_of_linkage_sets);
        /* Perform another round of Gene-pool Optimal Mixing with the donors randomly selected from the archive. */
        for(i = 0; i < number_of_linkage_sets; i++)
        {
            int population_kind;
            std::tie(donor_index, population_kind) = getForcedImprovementDonor(solution_idx, cluster_idx, -1, u_fi.data());

            char **ip_population;
            std::tie(std::ignore, ip_population, std::ignore, std::ignore,
                     std::ignore) = getPopulationKindAttributes(population_kind);

            linkage_group_index = order[i];
            copyValuesFromDonorToOffspring(result, ip_population[donor_index], cluster_idx, linkage_group_index);
            mutateSolution(result, linkage_group_index, cluster_idx);

            /* Check if the new intermediate solution is different from the previous state. */
            is_unchanged = TRUE;    
            for( j = 0; j < lt_number_of_indices[cluster_idx][linkage_group_index]; j++ )
            {
                if( backup[lt[cluster_idx][linkage_group_index][j]] != result[lt[cluster_idx][linkage_group_index][j]] )
                {
                    is_unchanged = FALSE;
                    break;
                }
            }           

            if( is_unchanged == FALSE )
            {
                is_improved = FALSE;

                evaluateIndividual(result, obj, con, metadata, NOT_EXTREME_CLUSTER);
                updateElitistArchiveWithReplacementOfExistedMember(result, obj, *con, metadata, &is_new_nondominated_point, &is_dominated_by_archive);

                /* Check for (strict) Pareto domination. */
                if ( constraintParetoDominates( obj, *con, obj_backup, con_backup) )
                    is_improved = TRUE;

                /* Check if a truly new non-dominated solution is created. */
                if(is_new_nondominated_point)
                    is_improved = TRUE;

                // Compute current scalarized Tschebysheff distance.
                // Update ideal point.
                updateIdealPointWithNewObjective(obj, reference);

                // Compute scalarized value (with the new )
                double tsch = computeTchebysheffDistance(number_of_objectives, obj, reference, tsch_weight_vector, ranges);
                double tsch_backup = computeTchebysheffDistance(number_of_objectives, obj_backup, reference, tsch_weight_vector, ranges);

                if ( tsch < tsch_backup )
                {
                    is_improved = true;
                }
                if ( is_improved )
                {
                    changed = TRUE;
                    improved = TRUE;
                    copyFromAToB(result, obj, *con, metadata,  backup, obj_backup, &con_backup, metadata_backup);
                    break;
                }
                else
                    copyFromAToB(backup, obj_backup, con_backup, metadata_backup, result, obj, con, metadata);
            }
        }
        free(order);

        if(!changed && fi_do_replacement)
        {
            int population_kind;
            std::tie(donor_index, population_kind) = getForcedImprovementReplacement(solution_idx, cluster_idx, -1, u_fi.data());

            char **ip_population;
            double **ip_objective_values, *ip_constraint_values;
            void **ip_solution_metadata;
            std::tie(std::ignore, ip_population, ip_objective_values, ip_constraint_values,
                     ip_solution_metadata) = getPopulationKindAttributes(population_kind);

            copyFromAToB(ip_population[donor_index], ip_objective_values[donor_index], 
                    ip_constraint_values[donor_index], ip_solution_metadata[donor_index],
                    result, obj, con, metadata);

            replaced = TRUE;
        }
    }

    if (
        (((reset_nis_on & 1) > 0) && changed) ||
        (((reset_nis_on & 2) > 0) && improved) ||
        (((reset_nis_on & 4) > 0) && replaced)
    )
        s_NIS[solution_idx] = 0;
    else
        s_NIS[solution_idx] += 1;

    free( backup ); free( obj_backup ); 
    cleanupSolutionMetadata(metadata_backup);
}

/**
 * HV-objective Gene-pool Optimal Mixing.
 * Accept a change if the hypervolume contribution increases.
 * Pseudo_objective_values is initially a copy of the objective values array (without copying the underlying pointers)
 * to allow for replacement in the HV computation, and potentially allow for easy reconfiguration towards steady state.
 */
void performRankHVObjectiveGenepoolOptimalMixing(
    int cluster_idx, int solution_idx, double** pseudo_objective_values,
    char *parent, double *parent_obj, double parent_con, void *parent_metadata,
    char *result, double        *obj, double       *con, void        *metadata)
{
    char   *backup, *donor, is_unchanged, changed, improved, is_improved, is_strictly_improved, replaced, is_new_nondominated_point, is_dominated_by_archive;
    int     i, j, donor_index, *order, linkage_group_index, number_of_linkage_sets;
    double  *obj_backup, con_backup;

    void *metadata_backup = NULL;
    initSolutionMetadata(metadata_backup);

    /* Clone the parent solution. */
    copyFromAToB(parent, parent_obj, parent_con, parent_metadata, result, obj, con, metadata);
    // Offspring is now initialized: update variable.
    initialized_offspring_size = solution_idx;

    /* Create a backup version of the parent solution. */
    backup = (char *) Malloc( number_of_parameters*sizeof( char ) );
    obj_backup = (double *) Malloc( number_of_objectives*sizeof( double ) );
    copyFromAToB(result, obj, *con, metadata, backup, obj_backup, &con_backup, metadata_backup);

    number_of_linkage_sets = lt_length[cluster_idx]; // Root is already removed from the tree - 1; /* Remove root from the linkage tree. */
    order = createRandomOrdering(number_of_linkage_sets);

    /* Determine the pool of indices in the population with which we can mix */
    std::vector<int> donor_indices;
    determineMixingPoolForSolution(solution_idx, cluster_idx, -1, donor_indices);

    if(donor_indices.size() == 0)
    {
        // Population has converged, exit early
        return;
    }

    // Compute the current contribution at the current rank of the solution.
    double backup_contribution;
    {
        // Alter the pseudo population
        pseudo_objective_values[solution_idx] = obj_backup;

        // Re-compute the current rank assignment if steady state.
        // This is required as changes of other population members may alter the rank of the current solution, which will either make
        // the acceptance criterion too strict (rank should be lower, got dominated by solution of previously the same rank)
        // or too lenient (rank should be higher, mostly makes acceptance too strict for other solutions, can lead to weird behavior)
        if (hv_steady_state)
        {
            rank.resize(population_size);
            rank_order.resize(0);
            max_rank = nondominated_sort(population_size, pseudo_objective_values, rank.data(), rank_order, rank_start_end);
            hv_per_rank.resize(max_rank);
        }

        // Compute HV contribution right now.
        computeHVForRank(adjusted_nadir, pseudo_objective_values, rank[solution_idx]);
        backup_contribution = hv_contributions_per_population_member[solution_idx];
    }

    /* Traverse the linkage tree for Gene-pool Optimal Mixing */
    changed = FALSE;
    improved = FALSE;
    replaced = FALSE;
    for( i = 0; i < number_of_linkage_sets; i++ )
    {
        linkage_group_index = order[i];

        is_unchanged = TRUE;
        for (int donor_indices_left = donor_indices.size(); donor_indices_left > 0; --donor_indices_left)
        {
            // Grab a random solution from the donor pool.
            int donor_indices_index = randomInt(donor_indices_left);
            donor_index = donor_indices[donor_indices_index];
            std::swap(donor_indices[donor_indices_index], donor_indices[donor_indices_left - 1]);
            donor = population[donor_index];

            copyValuesFromDonorToOffspring(result, donor, cluster_idx, linkage_group_index);     
            mutateSolution(result, linkage_group_index, cluster_idx);

            /* Check if the new intermediate solution is different from the previous state. */
            is_unchanged = TRUE;
            for( j = 0; j < lt_number_of_indices[cluster_idx][linkage_group_index]; j++ )
            {
                if( backup[lt[cluster_idx][linkage_group_index][j]] != result[lt[cluster_idx][linkage_group_index][j]] )
                {
                    is_unchanged = FALSE;
                    break;
                }
            }

            // Break if solution was changed.
            // Or donor search is not enabled
            if ( is_unchanged == FALSE || !use_donor_search ) break;
        }

        if( is_unchanged == FALSE )
        {
            is_improved = FALSE;
            is_strictly_improved = FALSE;
            evaluateIndividual(result, obj, con, metadata, NOT_EXTREME_CLUSTER);
            updateElitistArchiveWithReplacementOfExistedMember(result, obj, *con, metadata, &is_new_nondominated_point, &is_dominated_by_archive);

            /* Check for weak Pareto domination. */
            if ( constraintWeaklyParetoDominates( obj, *con, obj_backup, con_backup) )
                is_improved = TRUE;
            if ( constraintParetoDominates( obj, *con, obj_backup, con_backup) )
                is_strictly_improved = TRUE;

            /* Check if the new intermediate solution is dominated by any solution in the archive. 
                Note that if the new intermediate solution is a point in the archive, it is NOT considered to be dominated by the archive.*/
            if ( !is_dominated_by_archive )
                is_strictly_improved = TRUE;
            
            bool new_point_changed_adjusted_nadir = updateAdjustedNadir(optimization, utopian_point.data(), adjusted_nadir.data(), obj, number_of_objectives);
            if (new_point_changed_adjusted_nadir)
            {
                // Replace pointer in population such that we refer to the current solution.
                pseudo_objective_values[solution_idx] = obj_backup;
                // Unlike the ranked approach, this one does not use ranking.
                // Compute HV contribution right now.
                //adjusted_nadir, pseudo_objective_values, rank[solution_index]
                computeHVForRank(adjusted_nadir, pseudo_objective_values, rank[solution_idx]);
                backup_contribution = hv_contributions_per_population_member[solution_idx];
            }

            // Replace pointer in population such that we refer to the current solution.
            pseudo_objective_values[solution_idx] = obj;
            computeHVForRank(adjusted_nadir, pseudo_objective_values, rank[solution_idx]);
            double contribution = hv_contributions_per_population_member[solution_idx];

            // MAYBE: Allow equal change.
            if ( contribution >= backup_contribution)
                is_improved = TRUE;
            if ( contribution > backup_contribution)
                is_strictly_improved = TRUE;
            
            if ( is_strictly_improved )
                is_improved = TRUE;

            if ( is_improved )
            {
                changed = TRUE;
                copyFromAToB(result, obj, *con, metadata,  backup, obj_backup, &con_backup, metadata_backup);
                backup_contribution = contribution;
            }
            else
                copyFromAToB(backup, obj_backup, con_backup, metadata_backup, result, obj, con, metadata);
        
            if ( is_strictly_improved )
                improved = TRUE;
        }
    }

    free(order);

    /* Forced Improvement */
    if ((fi_mode >= 0) && ((!changed) || ((use_solution_NIS_instead_of_population_NIS
                                               ? (s_NIS[solution_idx] > (1 + floor(log10(population_size))))
                                               : (t_NIS > (1 + floor(log10(population_size))))))))
    {
        // Note, direction vector for the original objective is always the same for FI.
        // No need to recompute every iteration / every subset.
        std::vector<double> u_fi;
        if (FImodeRequiresDirection())
            computeNormalizedDirectionVector(u_fi, nadir_point.data(), obj, objective_ranges);

        changed = FALSE;
        order = createRandomOrdering(number_of_linkage_sets);
        /* Perform another round of Gene-pool Optimal Mixing with the donors randomly selected from the archive. */
        for(i = 0; i < number_of_linkage_sets; i++)
        {
            int population_kind;
            std::tie(donor_index, population_kind) = getForcedImprovementDonor(solution_idx, cluster_idx, -1, u_fi.data());

            char **ip_population;
            std::tie(std::ignore, ip_population, std::ignore, std::ignore,
                     std::ignore) = getPopulationKindAttributes(population_kind);

            linkage_group_index = order[i];
            copyValuesFromDonorToOffspring(result, ip_population[donor_index], cluster_idx, linkage_group_index);
            mutateSolution(result, linkage_group_index, cluster_idx);

            /* Check if the new intermediate solution is different from the previous state. */
            is_unchanged = TRUE;    
            for( j = 0; j < lt_number_of_indices[cluster_idx][linkage_group_index]; j++ )
            {
                if( backup[lt[cluster_idx][linkage_group_index][j]] != result[lt[cluster_idx][linkage_group_index][j]] )
                {
                    is_unchanged = FALSE;
                    break;
                }
            }           

            if( is_unchanged == FALSE )
            {
                is_improved = FALSE;

                evaluateIndividual(result, obj, con, metadata, NOT_EXTREME_CLUSTER);
                updateElitistArchiveWithReplacementOfExistedMember(result, obj, *con, metadata, &is_new_nondominated_point, &is_dominated_by_archive);

                /* Check for (strict) Pareto domination. */
                if ( constraintParetoDominates( obj, *con, obj_backup, con_backup) )
                    is_improved = TRUE;

                /* Check if a truly new non-dominated solution is created. */
                if(is_new_nondominated_point)
                    is_improved = TRUE;

                bool new_point_changed_adjusted_nadir = updateAdjustedNadir(optimization, utopian_point.data(), adjusted_nadir.data(), obj, number_of_objectives);
                if (new_point_changed_adjusted_nadir)
                {
                    // Replace pointer in population such that we refer to the current solution.
                    pseudo_objective_values[solution_idx] = obj_backup;
                    computeHVForRank(adjusted_nadir, pseudo_objective_values, rank[solution_idx]);
                    backup_contribution = hv_contributions_per_population_member[solution_idx];
                }

                // Replace pointer in population such that we refer to the current solution.
                pseudo_objective_values[solution_idx] = obj;
                computeHVForRank(adjusted_nadir, pseudo_objective_values, rank[solution_idx]);
                double contribution = hv_contributions_per_population_member[solution_idx];

                if (contribution > backup_contribution)
                    is_improved = TRUE;
                
                if ( is_improved )
                {
                    changed = TRUE;
                    improved = TRUE;
                    copyFromAToB(result, obj, *con, metadata,  backup, obj_backup, &con_backup, metadata_backup);
                    // Only here for completeness, but as this variable won't be read anymore this is hardly neccesary.
                    backup_contribution = contribution;
                    break;
                }
                else
                    copyFromAToB(backup, obj_backup, con_backup, metadata_backup, result, obj, con, metadata);
            }
        }
        free(order);

        if(!changed && fi_do_replacement)
        {
            int population_kind;
            std::tie(donor_index, population_kind) = getForcedImprovementReplacement(solution_idx, cluster_idx, -1, u_fi.data());

            char **ip_population;
            double **ip_objective_values;
            double *ip_constraint_values;
            void **ip_solution_metadata;
            std::tie(std::ignore, ip_population, ip_objective_values, ip_constraint_values,
                     ip_solution_metadata) = getPopulationKindAttributes(population_kind);


            copyFromAToB(ip_population[donor_index], ip_objective_values[donor_index], 
                    ip_constraint_values[donor_index], ip_solution_metadata[donor_index],
                    result, obj, con, metadata);

            replaced = TRUE;
        }
    }

    // Return pseudo_objective_values back to being a copy (as far as modifications in this function are concerned)
    // if we are not operating in a steady state style.
    if (!hv_steady_state)
        pseudo_objective_values[solution_idx] = objective_values[solution_idx];
    else
        // Use objective values instead.
        pseudo_objective_values[solution_idx] = obj;

    if (
        (((reset_nis_on & 1) > 0) && changed) ||
        (((reset_nis_on & 2) > 0) && improved) ||
        (((reset_nis_on & 4) > 0) && replaced)
    )
        s_NIS[solution_idx] = 0;
    else
        s_NIS[solution_idx] += 1;

    free( backup ); free( obj_backup );
    cleanupSolutionMetadata(metadata_backup);
}
/**
 * UHV-objective Gene-pool Optimal Mixing 
 * Accept a change if the UHVI indicator increases.
 * Pseudo_objective_values is initially a copy of the objective values array (without copying the underlying pointers).
 */
void performUHVIObjectiveGenepoolOptimalMixing(
    int cluster_idx, int solution_idx, double** pseudo_objective_values,
    char *parent, double *parent_obj, double parent_con, void *parent_metadata,
    char *result, double        *obj, double       *con, void        *metadata)
{
    char   *backup, *donor, is_unchanged, changed, improved, is_improved, is_strictly_improved, replaced, is_new_nondominated_point, is_dominated_by_archive;
    int     i, j, donor_index, *order, linkage_group_index, number_of_linkage_sets;
    double  *obj_backup, con_backup;

    void *metadata_backup = NULL;
    initSolutionMetadata(metadata_backup);

    /* Clone the parent solution. */
    copyFromAToB(parent, parent_obj, parent_con, parent_metadata, result, obj, con, metadata);
    // Offspring is now initialized: update variable.
    initialized_offspring_size = solution_idx;

    /* Create a backup version of the parent solution. */
    backup = (char *) Malloc( number_of_parameters*sizeof( char ) );
    obj_backup = (double *) Malloc( number_of_objectives*sizeof( double ) );
    copyFromAToB(result, obj, *con, metadata, backup, obj_backup, &con_backup, metadata_backup);

    number_of_linkage_sets = lt_length[cluster_idx]; // Root is already removed from the tree - 1; /* Remove root from the linkage tree. */
    order = createRandomOrdering(number_of_linkage_sets);

    /* Determine the pool of indices in the population with which we can mix */
    std::vector<int> donor_indices;
    determineMixingPoolForSolution(solution_idx, cluster_idx, -1, donor_indices);

    if(donor_indices.size() == 0)
    {
        // Population has converged, exit early
        return;
    }

    std::vector<int> hv_indices;

    // Compute the current contribution at the current rank of the solution.
    double backup_contribution;
    {
        if (hv_improvement_subset)
        {
            hv_indices = donor_indices;
        }
        else
        {
            hv_indices.resize(population_size);
            std::iota(hv_indices.begin(), hv_indices.end(), 0);
        } 
        // Alter the pseudo population
        pseudo_objective_values[solution_idx] = obj_backup;
        // Unlike the ranked approach, this one does not use ranking.
        // Compute HV contribution right now.
        //adjusted_nadir, pseudo_objective_values, rank[solution_index]
        compute2DUncrowdedHypervolumeSubsetContributions(
            &solution_idx, 1, 
            hv_indices.data(), hv_indices.size(),
            optimization, adjusted_nadir.data(), pseudo_objective_values, 
            hv_contributions_per_population_member.data(),
            hv_density_correction, objective_range.data());
        backup_contribution = hv_contributions_per_population_member[solution_idx];
    }

    /* Traverse the linkage tree for Gene-pool Optimal Mixing */
    changed = FALSE;
    improved = FALSE;
    replaced = FALSE;
    for( i = 0; i < number_of_linkage_sets; i++ )
    {
        linkage_group_index = order[i];

        is_unchanged = TRUE;
        for (int donor_indices_left = donor_indices.size(); donor_indices_left > 0; --donor_indices_left)
        {
            // Grab a random solution from the donor pool.
            int donor_indices_index = randomInt(donor_indices_left);
            donor_index = donor_indices[donor_indices_index];
            std::swap(donor_indices[donor_indices_index], donor_indices[donor_indices_left - 1]);
            donor = population[donor_index];

            copyValuesFromDonorToOffspring(result, donor, cluster_idx, linkage_group_index);     
            mutateSolution(result, linkage_group_index, cluster_idx);

            /* Check if the new intermediate solution is different from the previous state. */
            is_unchanged = TRUE;
            for( j = 0; j < lt_number_of_indices[cluster_idx][linkage_group_index]; j++ )
            {
                if( backup[lt[cluster_idx][linkage_group_index][j]] != result[lt[cluster_idx][linkage_group_index][j]] )
                {
                    is_unchanged = FALSE;
                    break;
                }
            }

            // Break if solution was changed.
            // Or donor search is not enabled
            if ( is_unchanged == FALSE || !use_donor_search ) break;
        }

        if( is_unchanged == FALSE )
        {
            is_improved = FALSE;
            is_strictly_improved = FALSE;
            evaluateIndividual(result, obj, con, metadata, NOT_EXTREME_CLUSTER);
            updateElitistArchiveWithReplacementOfExistedMember(result, obj, *con, metadata, &is_new_nondominated_point, &is_dominated_by_archive);

            /* Check for weak Pareto domination. */
            if ( constraintWeaklyParetoDominates( obj, *con, obj_backup, con_backup) )
                is_improved = TRUE;
            if ( constraintParetoDominates( obj, *con, obj_backup, con_backup) )
                is_strictly_improved = TRUE;

            /* Check if the new intermediate solution is dominated by any solution in the archive. 
                Note that if the new intermediate solution is a point in the archive, it is NOT considered to be dominated by the archive.*/
            // if ( !is_dominated_by_archive )
            //    is_improved = TRUE;

            bool new_point_changed_adjusted_nadir = updateAdjustedNadir(optimization, utopian_point.data(), adjusted_nadir.data(), obj, number_of_objectives);
            if (new_point_changed_adjusted_nadir)
            {
                // Replace pointer in population such that we refer to the current solution.
                pseudo_objective_values[solution_idx] = obj_backup;
                // Unlike the ranked approach, this one does not use ranking.
                // Compute HV contribution right now.
                //adjusted_nadir, pseudo_objective_values, rank[solution_index]
                compute2DUncrowdedHypervolumeSubsetContributions(
                    &solution_idx, 1, 
                    hv_indices.data(), hv_indices.size(),
                    optimization, adjusted_nadir.data(), pseudo_objective_values, 
                    hv_contributions_per_population_member.data(), 
                    hv_density_correction, objective_range.data());
                backup_contribution = hv_contributions_per_population_member[solution_idx];
            }

            // Replace pointer in population such that we refer to the current solution.
            pseudo_objective_values[solution_idx] = obj;
            // Unlike the ranked approach, this one does not use ranking.
            // Compute HV contribution right now.
            //adjusted_nadir, pseudo_objective_values, rank[solution_index]
            compute2DUncrowdedHypervolumeSubsetContributions(
                &solution_idx, 1, 
                hv_indices.data(), hv_indices.size(),
                optimization, adjusted_nadir.data(), pseudo_objective_values, 
                hv_contributions_per_population_member.data(),
                hv_density_correction, objective_range.data());
            double contribution = hv_contributions_per_population_member[solution_idx];

            if ( contribution > backup_contribution)
                is_strictly_improved = TRUE;
            if ( contribution >= backup_contribution)
                is_improved = TRUE;

            if ( is_strictly_improved )
                is_improved = TRUE;
            

            if ( is_improved )
            {
                changed = TRUE;
                copyFromAToB(result, obj, *con, metadata,  backup, obj_backup, &con_backup, metadata_backup);
                backup_contribution = contribution;
            }
            else
                copyFromAToB(backup, obj_backup, con_backup, metadata_backup, result, obj, con, metadata);
        
            if ( is_strictly_improved )
                improved = TRUE;
        }
    }

    free(order);

    /* Forced Improvement */
    if ((fi_mode >= 0) && ((!changed) || ((use_solution_NIS_instead_of_population_NIS
                                               ? (s_NIS[solution_idx] > (1 + floor(log10(population_size))))
                                               : (t_NIS > (1 + floor(log10(population_size))))))))
    {
        // Note, direction vector for the original objective is always the same for FI.
        // No need to recompute every iteration / every subset.
        std::vector<double> u_fi;
        if (FImodeRequiresDirection())
            computeNormalizedDirectionVector(u_fi, nadir_point.data(), obj, objective_ranges);

        // Switch to full-population for FI
        if (hv_improvement_subset && !hv_improvement_subset_fi)
        {
            hv_indices.resize(population_size);
            std::iota(hv_indices.begin(), hv_indices.end(), 0);

            compute2DUncrowdedHypervolumeSubsetContributions(
                &solution_idx, 1, 
                hv_indices.data(), hv_indices.size(),
                optimization, adjusted_nadir.data(), pseudo_objective_values, 
                hv_contributions_per_population_member.data(),
                hv_density_correction, objective_range.data());
            backup_contribution = hv_contributions_per_population_member[solution_idx];
        }
        changed = FALSE;
        order = createRandomOrdering(number_of_linkage_sets);
        /* Perform another round of Gene-pool Optimal Mixing with the donors randomly selected from the archive. */
        for(i = 0; i < number_of_linkage_sets; i++)
        {
            int population_kind;
            std::tie(donor_index, population_kind) = getForcedImprovementDonor(solution_idx, cluster_idx, -1, u_fi.data());

            char **ip_population;
            std::tie(std::ignore, ip_population, std::ignore, std::ignore,
                     std::ignore) = getPopulationKindAttributes(population_kind);

            linkage_group_index = order[i];
            copyValuesFromDonorToOffspring(result, ip_population[donor_index], cluster_idx, linkage_group_index);
            mutateSolution(result, linkage_group_index, cluster_idx);

            /* Check if the new intermediate solution is different from the previous state. */
            is_unchanged = TRUE;    
            for( j = 0; j < lt_number_of_indices[cluster_idx][linkage_group_index]; j++ )
            {
                if( backup[lt[cluster_idx][linkage_group_index][j]] != result[lt[cluster_idx][linkage_group_index][j]] )
                {
                    is_unchanged = FALSE;
                    break;
                }
            }           

            if( is_unchanged == FALSE )
            {
                is_improved = FALSE;

                evaluateIndividual(result, obj, con, metadata, NOT_EXTREME_CLUSTER);
                updateElitistArchiveWithReplacementOfExistedMember(result, obj, *con, metadata, &is_new_nondominated_point, &is_dominated_by_archive);

                /* Check for (strict) Pareto domination. */
                if ( constraintParetoDominates( obj, *con, obj_backup, con_backup) )
                    is_improved = TRUE;

                /* Check if a truly new non-dominated solution is created. */
                if(is_new_nondominated_point)
                    is_improved = TRUE;

                bool new_point_changed_adjusted_nadir = updateAdjustedNadir(optimization, utopian_point.data(), adjusted_nadir.data(), obj, number_of_objectives);
                if (new_point_changed_adjusted_nadir)
                {
                    // Replace pointer in population such that we refer to the current solution.
                    pseudo_objective_values[solution_idx] = obj_backup;
                    // Unlike the ranked approach, this one does not use ranking.
                    // Compute HV contribution right now.
                    //adjusted_nadir, pseudo_objective_values, rank[solution_index]
                    compute2DUncrowdedHypervolumeSubsetContributions(
                        &solution_idx, 1,
                        hv_indices.data(), hv_indices.size(),
                        optimization, adjusted_nadir.data(), pseudo_objective_values, 
                        hv_contributions_per_population_member.data(), 
                        hv_density_correction, objective_range.data());
                    backup_contribution = hv_contributions_per_population_member[solution_idx];
                }

                // Replace pointer in population such that we refer to the current solution.
                pseudo_objective_values[solution_idx] = obj;
                // Unlike the ranked approach, this one does not use ranking.
                // Compute HV contribution right now.
                //adjusted_nadir, pseudo_objective_values, rank[solution_index]
                compute2DUncrowdedHypervolumeSubsetContributions(
                    &solution_idx, 1,
                    hv_indices.data(), hv_indices.size(),
                    optimization, adjusted_nadir.data(), pseudo_objective_values, 
                    hv_contributions_per_population_member.data(),
                    hv_density_correction, objective_range.data());
                double contribution = hv_contributions_per_population_member[solution_idx];

                if (contribution > backup_contribution)
                    is_improved = TRUE;
                
                if ( is_improved )
                {
                    changed = TRUE;
                    improved = TRUE;
                    copyFromAToB(result, obj, *con, metadata,  backup, obj_backup, &con_backup, metadata_backup);
                    // Only here for completeness, but as this variable won't be read anymore this is hardly neccesary.
                    backup_contribution = contribution;
                    break;
                }
                else
                    copyFromAToB(backup, obj_backup, con_backup, metadata_backup, result, obj, con, metadata);
            }
        }
        free(order);

        if(!changed && fi_do_replacement)
        {
            int population_kind;
            std::tie(donor_index, population_kind) = getForcedImprovementReplacement(solution_idx, cluster_idx, -1, u_fi.data());

            char **ip_population;
            double **ip_objective_values;
            double *ip_constraint_values;
            void **ip_solution_metadata;
            std::tie(std::ignore, ip_population, ip_objective_values, ip_constraint_values,
                     ip_solution_metadata) = getPopulationKindAttributes(population_kind);

            copyFromAToB(ip_population[donor_index], ip_objective_values[donor_index], 
                    ip_constraint_values[donor_index], ip_solution_metadata[donor_index],
                    result, obj, con, metadata);

            replaced = TRUE;
        }
    }

    // Return pseudo_objective_values back to being a copy (as far as modifications in this function are concerned)
    // if we are not operating in a steady state style.
    if (!hv_steady_state)
        pseudo_objective_values[solution_idx] = objective_values[solution_idx];
    else
        // Use objective values instead.
        pseudo_objective_values[solution_idx] = obj;

    if (
        (((reset_nis_on & 1) > 0) && changed) ||
        (((reset_nis_on & 2) > 0) && improved) ||
        (((reset_nis_on & 4) > 0) && replaced)
    )
        s_NIS[solution_idx] = 0;
    else
        s_NIS[solution_idx] += 1;

    free( backup ); free( obj_backup );
    cleanupSolutionMetadata(metadata_backup);
}

/**
 * Determines the solutions that finally survive the generation (offspring only).
 */
void selectFinalSurvivors()
{
    int i, j;

    for( i = 0; i < population_size; i++ )
    {
        for( j = 0; j < number_of_parameters; j++ )
            population[i][j] = offspring[i][j];
        for( j = 0; j < number_of_objectives; j++)
            objective_values[i][j]  = objective_values_offspring[i][j];
        constraint_values[i] = constraint_values_offspring[i];

        copySolutionMetadata(solution_metadata_offspring[i], solution_metadata[i]);
    }
}

void freeAuxiliaryPopulations()
{
    int i, k;

    if(population_indices_of_cluster_members != NULL)
    {
        for(k = 0; k < number_of_mixing_components; k++)
            free(population_indices_of_cluster_members[k]);
        free(population_indices_of_cluster_members);
        population_indices_of_cluster_members = NULL;
        free(population_cluster_sizes);
    }

    if(offspring != NULL)
    {
        // Offspring is going to be uninitialized. Reset back to 0.
        initialized_offspring_size = 0;
        for(i = 0; i < offspring_size; i++)
        {
            free(offspring[i]);
            free(objective_values_offspring[i]);
            cleanupSolutionMetadata(solution_metadata_offspring[i]);
        }
        free(offspring);
        free(objective_values_offspring);
        free(constraint_values_offspring);
        free(solution_metadata_offspring);
        offspring = NULL;
    }
    
    ezilaitiniClusters();
    
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Parameter-free Mechanism -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
void prepareSpecialClusteringMode()
{
    switch (arg_num_cluster_mode)
    {
        case 's':
            // As many clusters such that
            // 2|P| / num_clust = sqrt(|P|)
            smallest_number_of_clusters = std::ceil(std::sqrt(smallest_population_size));
        break;
}
    }
int getClusterSizeForPopulation(int population_index)
{
    switch (arg_num_cluster_mode)
    {
        case 's':
            {
            // As many clusters such that
            // 2|P| / num_clust = sqrt(|P|)
            int out = std::ceil(std::sqrt(array_of_population_sizes[population_index]));
            return out;
            }
        break;
        default:
            if (fixed_number_of_clusters)
                return array_of_number_of_clusters[population_index-1];
            else
                return array_of_number_of_clusters[population_index-1] + 1;
    }
}

void getPopulationSizingProperties()
{
    maximum_number_of_populations = 20;
    allow_starting_of_more_than_one_population = true;
    if (arg_population_size == -1)
    {
        smallest_population_size = 8;
    }
    else
    {
        smallest_population_size = arg_population_size;
        allow_starting_of_more_than_one_population = false;
        maximum_number_of_populations = 1;
    }
    if (arg_num_clusters == 0)
    {
        smallest_number_of_clusters = number_of_objectives + 1;
    }
    else if (arg_num_cluster_mode != 0)
    {
        prepareSpecialClusteringMode();
    }
    else
    {
        smallest_number_of_clusters = std::abs(arg_num_clusters);
        fixed_number_of_clusters = arg_num_clusters > 0;
    }
}

void initializeMemoryForArrayOfPopulations()
{
    int i;
    getPopulationSizingProperties();

    array_of_populations                = (char***)Malloc(maximum_number_of_populations*sizeof(char**));
    array_of_objective_values           = (double***)Malloc(maximum_number_of_populations*sizeof(double**));
    array_of_constraint_values          = (double**)Malloc(maximum_number_of_populations*sizeof(double*));
    array_of_objective_ranges           = (double**)Malloc(maximum_number_of_populations*sizeof(double));

    array_of_solution_metadatas       = (void***)Malloc(maximum_number_of_populations*sizeof(void*));
    array_of_solution_k               = (int**)Malloc(maximum_number_of_populations*sizeof(int*));

    // For use with scalarization.
    if (use_scalarization)
        array_of_tsch_weight_vectors    = (double***)Malloc(maximum_number_of_populations*sizeof(double**));
    else
        array_of_tsch_weight_vectors    = NULL;

    array_of_s_NIS                      = (int**)Malloc(maximum_number_of_populations*sizeof(int*));
    array_of_t_NIS                      = (int*)Malloc(maximum_number_of_populations*sizeof(int));
    array_of_number_of_generations                = (int*)Malloc(maximum_number_of_populations*sizeof(int));
    for(i = 0; i < maximum_number_of_populations; i++)
    {
        array_of_number_of_generations[i]         = 0;
        array_of_t_NIS[i]               = 0;
        array_of_s_NIS[i] = NULL;
    }

    array_of_number_of_evaluations_per_population = (long*)Malloc(maximum_number_of_populations*sizeof(long));
    for(i = 0; i < maximum_number_of_populations; i++)
        array_of_number_of_evaluations_per_population[i] = 0;

    /* Popupulation-sizing free scheme. */
    array_of_population_sizes           = (int*)Malloc(maximum_number_of_populations*sizeof(int));
    array_of_population_sizes[0]        = smallest_population_size;
    for(i = 1; i < maximum_number_of_populations; i++)
        array_of_population_sizes[i]    = array_of_population_sizes[i-1]*2;

    /* Number-of-clusters parameter-free scheme. */
    array_of_number_of_clusters         = (int*)Malloc(maximum_number_of_populations*sizeof(int));
    array_of_number_of_clusters[0]      = smallest_number_of_clusters;
    for(i = 1; i < maximum_number_of_populations; i++)
        array_of_number_of_clusters[i] = getClusterSizeForPopulation(i);
    
}

void putInitializedPopulationIntoArray()
{
    array_of_objective_ranges[population_id]    = objective_ranges;
    array_of_populations[population_id]         = population;
    array_of_objective_values[population_id]    = objective_values;
    array_of_constraint_values[population_id]   = constraint_values;
    array_of_t_NIS[population_id]               = 0;
    array_of_s_NIS[population_id]               = s_NIS;

    array_of_solution_metadatas[population_id]  = solution_metadata;
    array_of_solution_k[population_id]          = solution_k;
    
    if (use_scalarization)
        array_of_tsch_weight_vectors[population_id] = tsch_weight_vectors;
}

void assignPointersToCorrespondingPopulation()
{
    population                  = array_of_populations[population_id];
    objective_values            = array_of_objective_values[population_id];
    constraint_values           = array_of_constraint_values[population_id];
    population_size             = array_of_population_sizes[population_id];
    initialized_population_size = population_size;
    // Even if initialized, offspring does not belong to current population.
    initialized_offspring_size  = 0;
    objective_ranges            = array_of_objective_ranges[population_id];
    t_NIS                       = array_of_t_NIS[population_id];
    number_of_generations       = array_of_number_of_generations[population_id];
    number_of_mixing_components = array_of_number_of_clusters[population_id];
    orig_number_of_mixing_components = -1;

    solution_metadata           = array_of_solution_metadatas[population_id];
    solution_k                  = array_of_solution_k[population_id];
    s_NIS                       = array_of_s_NIS[population_id];

    if (use_scalarization)
        tsch_weight_vectors     = array_of_tsch_weight_vectors[population_id];
}

void ezilaitiniMemoryOfCorrespondingPopulation()
{
    int i;

    for( i = 0; i < population_size; i++ )
    {
        free( population[i] );
        free( objective_values[i] );

        cleanupSolutionMetadata( solution_metadata[i] );

        if (use_scalarization)
            free( tsch_weight_vectors[i] );
    }
    free( population );
    free( objective_values );
    free( constraint_values );
    free( objective_ranges );
    free( s_NIS );
    
    free( solution_k );

    free( solution_metadata );

    if (use_scalarization)
        free( tsch_weight_vectors );
}

void ezilaitiniArrayOfPopulation()
{
    int i;
    for(i = 0; i < number_of_populations; i++)
    {
        population_id = i;
        assignPointersToCorrespondingPopulation();
        ezilaitiniMemoryOfCorrespondingPopulation();
    }
    free(array_of_populations);
    free(array_of_objective_values);
    free(array_of_constraint_values);
    free(array_of_population_sizes);
    free(array_of_objective_ranges);
    free(array_of_t_NIS);
    free(array_of_number_of_generations);
    free(array_of_number_of_evaluations_per_population);
    free(array_of_number_of_clusters);
    free(array_of_s_NIS);
    free(array_of_solution_k);

    if (use_scalarization)
        free( array_of_tsch_weight_vectors );
}
/**
 * Schedule the run of multiple populations.
 */
void schedule_runMultiplePop_clusterPop_learnPop_improvePop()
{
    int i;

    initializeMemoryForArrayOfPopulations();
    initializeArrayOfParetoFronts();
    while( !checkTerminationCondition() )
    {
        population_id = 0;
        do
        {
            // Check `array_of_population_statuses` if not all have converged
            // or are to be skipped.
            if (population_id >= maximum_number_of_populations)
            {
                // std::cout << "Hit largest population size " << population_id << "/" << maximum_number_of_populations << std::endl;
                break;
            }
            if( array_of_number_of_generations[population_id] == 0 )
            {
                if (!allow_starting_of_more_than_one_population && population_id != 0)
                    break;

                population_size = array_of_population_sizes[population_id];
                initialized_population_size = 0;
                number_of_mixing_components = array_of_number_of_clusters[population_id];
                orig_number_of_mixing_components = -1;
                initialize();

                putInitializedPopulationIntoArray();

                if(stop_population_when_front_is_covered)
                {
                    updateParetoFrontForCurrentPopulation(objective_values, constraint_values, population_size);
                    checkWhichSmallerPopulationsNeedToStop();
                }

                if (population_id == 0)
                {
                    // Determine the reference point for computing hypervolume.
                    determineReferencePointForCurrentProblemHV();
                }

                writeGenerationalStatistics();
                freeAuxiliaryPopulations();
            }
            else if(array_of_population_statuses[population_id] == TRUE)
            {
                // Odd generation.
                assignPointersToCorrespondingPopulation();

                if (approach_mode == 1 || approach_mode == 2)
                {
                    if (use_original_clustering_with_domination_iteration)
                    {
                        performClustering(clustering_mode);
                    }
                    else if (use_knn_per_lineinstead_of_assigned_line)
                    {
                        performClustering(2);
                    }
                    else
                    {
                        performClustering(1);
                    }

                    // writeCurrentPopulation();
                    // writeCurrentLines();

                    learnLinkageOnCurrentPopulation();

                    computeObjectiveMinMaxRangesNadirAndUtopianPoint();

                    performPrecomputationForMixingPool();

                    if (use_single_objective_directions)
                    {
                        // With single objective
                        improveCurrentPopulation(0);
                    }
                    else
                    {
                        // Without single objective
                        improveCurrentPopulation(2);
                    }

                    selectFinalSurvivors();

                    computeObjectiveRanges();

                    adaptObjectiveDiscretization();

                    array_of_t_NIS[population_id] = t_NIS;

                    if(stop_population_when_front_is_covered)
                    {
                        updateParetoFrontForCurrentPopulation(objective_values, constraint_values, population_size);
                        checkWhichSmallerPopulationsNeedToStop();
                    }

                    writeGenerationalStatistics();
                    freeAuxiliaryPopulations();

                }
                else if (approach_mode == 3 || approach_mode == 4 || approach_mode == 5)
                {
                    // Skip domination based generation: No domination based steps are performed.
                }
                else
                {
                    if (domination_mode_uses_clustering_mode)
                        performClustering(clustering_mode);
                    else
                        performClustering(domination_based_clustering_mode);
                    learnLinkageOnCurrentPopulation();
                    computeObjectiveMinMaxRangesNadirAndUtopianPoint();

                    performPrecomputationForMixingPool();

                    if (use_scalarization)
                        improveCurrentPopulation(-1);
                    else
                        improveCurrentPopulation(0);

                    selectFinalSurvivors();

                    computeObjectiveRanges();

                    adaptObjectiveDiscretization();

                    array_of_t_NIS[population_id] = t_NIS;

                    if(stop_population_when_front_is_covered)
                    {
                        updateParetoFrontForCurrentPopulation(objective_values, constraint_values, population_size);
                        checkWhichSmallerPopulationsNeedToStop();
                    }

                    writeGenerationalStatistics();
                    freeAuxiliaryPopulations();

                }
                
                if (approach_mode == 1 || approach_mode == 2 || approach_mode == 3)
                {
                    // performClustering(clustering_mode);

                    if (use_knn_per_lineinstead_of_assigned_line)
                    {
                        performClustering(2);
                    }
                    else
                    {
                        performClustering(1);
                    }

                    learnLinkageOnCurrentPopulation();
                    // if (clustering_mode != 1) {
                    //     determineLines();
                    //     computeNearestLineForEachIndividual();
                    // }
                    // writeCurrentPopulation();
                    // writeCurrentLines();

                    computeObjectiveMinMaxRangesNadirAndUtopianPoint();

                    performPrecomputationForMixingPool();

                    // Requires lines to be prepared.
                    if (approach_mode == 2)
                        { determineORPs(); improveCurrentPopulation(3); }
                    else
                    {
                        
                        improveCurrentPopulation(1);
                    }

                    selectFinalSurvivors();

                    computeObjectiveRanges();

                    adaptObjectiveDiscretization();

                    array_of_t_NIS[population_id] = t_NIS;

                    if(stop_population_when_front_is_covered)
                    {
                        updateParetoFrontForCurrentPopulation(objective_values, constraint_values, population_size);
                        checkWhichSmallerPopulationsNeedToStop();
                    }

                    writeGenerationalStatistics();
                    freeAuxiliaryPopulations();
                }
                else if (approach_mode == 4 || approach_mode == 6)
                {
                    determineRanks();
                    computeObjectiveMinMaxRangesNadirAndUtopianPoint();
                    computeHVPerRank();

                    performClustering(clustering_mode);
                    learnLinkageOnCurrentPopulation();
                    
                    performPrecomputationForMixingPool();

                    improveCurrentPopulation(-2);

                    selectFinalSurvivors();

                    computeObjectiveRanges();

                    adaptObjectiveDiscretization();

                    array_of_t_NIS[population_id] = t_NIS;

                    if(stop_population_when_front_is_covered)
                    {
                        updateParetoFrontForCurrentPopulation(objective_values, constraint_values, population_size);
                        checkWhichSmallerPopulationsNeedToStop();
                    }

                    writeGenerationalStatistics();
                    freeAuxiliaryPopulations();
                }
                else if (approach_mode == 5 || approach_mode == 7)
                {
                    computeObjectiveMinMaxRangesNadirAndUtopianPoint();
                    performClustering(clustering_mode);
                    learnLinkageOnCurrentPopulation();
                    
                    performPrecomputationForMixingPool();

                    hv_contributions_per_population_member.resize(population_size);
                    improveCurrentPopulation(-3);

                    selectFinalSurvivors();

                    computeObjectiveRanges();

                    adaptObjectiveDiscretization();

                    array_of_t_NIS[population_id] = t_NIS;

                    if(stop_population_when_front_is_covered)
                    {
                        updateParetoFrontForCurrentPopulation(objective_values, constraint_values, population_size);
                        checkWhichSmallerPopulationsNeedToStop();
                    }

                    writeGenerationalStatistics();
                    freeAuxiliaryPopulations();
                }
            
                if (checkPopulationTerminationCriterion())
                {
                    array_of_population_statuses[population_id] = FALSE;
                }
            }

            
            
            array_of_number_of_generations[population_id]++;
            if(use_print_progress_to_screen)
                printf("%d ", array_of_number_of_generations[population_id]);
            population_id++;
            if(checkTerminationCondition() == TRUE)
                break;
        } while(array_of_number_of_generations[population_id-1] % generation_base == 0);
        if(use_print_progress_to_screen)
            printf(":   %ld\n", number_of_evaluations);
    }
    
    if(use_print_progress_to_screen)
    {
        printf("Population Status:\n");
        for(i=0; i < number_of_populations; i++)
            printf("Pop %d: %d\n", ((int)(pow(2,i)))*smallest_population_size, array_of_population_statuses[i]);
    }
    logNumberOfEvaluationsAtVTR();
    // Write last obtained hypervolume (in case of vtr, this will probably be 1!)
    // As point in time picks the last archive improvement: allows the file to be used as hitting time as well.
    writeCurrentHypervolume( true );

    writeCurrentElitistArchive( TRUE );
    writeCurrentLines( TRUE );
    writeCurrentPopulation( TRUE );
    ezilaitiniArrayOfPopulation();
    ezilaitiniArrayOfParetoFronts();
}

void schedule()
{   
    schedule_runMultiplePop_clusterPop_learnPop_improvePop();
}

/*---------------------Section Stop Smaller Populations -----------------------------------*/
void initializeArrayOfParetoFronts()
{
    int i;
    
    array_of_population_statuses                    = (char*)Malloc(maximum_number_of_populations*sizeof(char));
    for(i = 0; i < maximum_number_of_populations; i++)
        array_of_population_statuses[i] = TRUE;
    
    array_of_Pareto_front_size_of_each_population   = (int*)Malloc(maximum_number_of_populations*sizeof(int));
    for(i = 0; i < maximum_number_of_populations; i++)
        array_of_Pareto_front_size_of_each_population[i] = 0;

    array_of_Pareto_front_of_each_population        = (double***)Malloc(maximum_number_of_populations*sizeof(double**));
    for(i = 0; i < maximum_number_of_populations; i++)
        array_of_Pareto_front_of_each_population[i] = NULL;
}

void ezilaitiniArrayOfParetoFronts()
{
    int i, j;

    // FILE *file;
    // file = fopen("population_status.dat", "w");
    // for(i = 0; i < number_of_populations; i++)
    // {
    //     fprintf(file, "Pop %d: %d\n", ((int)(pow(2,i)))*smallest_population_size, array_of_population_statuses[i]);
    // }
    // fclose(file);

    for(i = 0; i < maximum_number_of_populations; i++)
    {
        if(array_of_Pareto_front_size_of_each_population[i] > 0)
        {
            for(j = 0; j < array_of_Pareto_front_size_of_each_population[i]; j++)
                free(array_of_Pareto_front_of_each_population[i][j]);
            free(array_of_Pareto_front_of_each_population[i]);
        }
    }
    free(array_of_Pareto_front_of_each_population);
    free(array_of_Pareto_front_size_of_each_population);
    free(array_of_population_statuses);
}

char checkParetoFrontCover(int pop_index_1, int pop_index_2)
{
    int i, j, count;
    count = 0;
    
    for(i = 0; i < array_of_Pareto_front_size_of_each_population[pop_index_2]; i++)
    {
        for(j = 0; j < array_of_Pareto_front_size_of_each_population[pop_index_1]; j++)
            if((constraintParetoDominates(array_of_Pareto_front_of_each_population[pop_index_1][j], 0, 
                array_of_Pareto_front_of_each_population[pop_index_2][i], 0) == TRUE) ||
                sameObjectiveBox(array_of_Pareto_front_of_each_population[pop_index_1][j], array_of_Pareto_front_of_each_population[pop_index_2][i]) == TRUE)
        {
            count++;
            break;
        }
    }
    // Check if all points in front 2 are dominated by or exist in front 1
    if(count == array_of_Pareto_front_size_of_each_population[pop_index_2])
        return TRUE;
    return FALSE;
}

void checkWhichSmallerPopulationsNeedToStop()
{
    int i;
    for(i = population_id - 1; i >= 0; i--)
    {
        if(array_of_population_statuses[i] == FALSE)
            continue;
        if(checkParetoFrontCover(population_id, i) == TRUE)
            array_of_population_statuses[i] = FALSE;
    }
}

void updateParetoFrontForCurrentPopulation(double **objective_values_pop, double *constraint_values_pop, int pop_size)
{
    int i, j, index, rank0_size;
    char *isDominated;
    isDominated = (char*)Malloc(pop_size*sizeof(char));
    for(i = 0; i < pop_size; i++)
        isDominated[i] = FALSE;
    for (i = 0; i < pop_size; i++)
    {
        if(isDominated[i] == TRUE)
            continue;
        for(j = i+1; j < pop_size; j++)
        {
            if(isDominated[j] == TRUE)
                continue;
            if(constraintParetoDominates(objective_values_pop[i], constraint_values_pop[i], objective_values_pop[j],constraint_values_pop[j]) == TRUE)
                isDominated[j]=TRUE;
            else if(constraintParetoDominates(objective_values_pop[j], constraint_values_pop[j], objective_values_pop[i],constraint_values_pop[i]) == TRUE)
            {
                isDominated[i]=TRUE;
                break;
            }
        }
    }

    rank0_size = 0;
    for(i = 0; i < pop_size; i++)
        if(isDominated[i]==FALSE)
            rank0_size++;

    if(array_of_Pareto_front_size_of_each_population[population_id] > 0)
    {
        for(i = 0; i < array_of_Pareto_front_size_of_each_population[population_id]; i++)
        {
            free(array_of_Pareto_front_of_each_population[population_id][i]);
        }
        free(array_of_Pareto_front_of_each_population[population_id]);        
    }

    array_of_Pareto_front_of_each_population[population_id] = (double**)Malloc(rank0_size*sizeof(double*));
    for(i = 0; i < rank0_size; i++)
        array_of_Pareto_front_of_each_population[population_id][i] = (double*)Malloc(number_of_objectives*sizeof(double));
    array_of_Pareto_front_size_of_each_population[population_id] = rank0_size;

    index = 0;
    for(i = 0; i < pop_size; i++)
    {
        if(isDominated[i] == TRUE)
            continue;
        for(j = 0; j < number_of_objectives; j++)
            array_of_Pareto_front_of_each_population[population_id][index][j] = objective_values_pop[i][j];
        index++;
    }
    free(isDominated);
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Run -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
void initializeCommonVariables()
{
    int i;

    initializeRandomNumberGenerator();
    generation_base                         = 2;

    number_of_generations                   = 0;
    number_of_evaluations                   = 0;
    objective_discretization_in_effect      = 0;
    elitist_archive_size                    = 0;
    elitist_archive_capacity                = 10;
    elitist_archive                         = (char **) Malloc( elitist_archive_capacity*sizeof( char * ) );
    elitist_archive_objective_values        = (double **) Malloc( elitist_archive_capacity*sizeof( double * ) );
    elitist_archive_constraint_values       = (double *) Malloc( elitist_archive_capacity*sizeof( double ) );
    
    elitist_archive_solution_metadata       = (void**) Malloc( elitist_archive_capacity*sizeof( void* ) );

    for( i = 0; i < elitist_archive_capacity; i++ )
    {
        elitist_archive[i]                  = (char *) Malloc( number_of_parameters*sizeof( char ) );
        elitist_archive_objective_values[i] = (double *) Malloc( number_of_objectives*sizeof( double ) );
        initSolutionMetadata(elitist_archive_solution_metadata[i]);
    }
    elitist_archive_copy                    = NULL;
    objective_discretization = (double *) Malloc( number_of_objectives*sizeof( double ) );

    MI_matrix = (double **) Malloc( number_of_parameters*sizeof( double * ) );
    for( i = 0; i < number_of_parameters; i++ )
        MI_matrix[i] = (double *) Malloc( number_of_parameters*sizeof( double ) );

    population_indices_of_cluster_members   = NULL;
    population_cluster_sizes                = NULL;

    offspring = NULL;
    
    number_of_populations = 0;

    lt = NULL;
}

void ezilaitiniCommonVariables( void )
{
    int      i;
    
    if( elitist_archive_copy != NULL )
    {
        for( i = 0; i < elitist_archive_copy_size; i++ )
        {
            free( elitist_archive_copy[i] );
            free( elitist_archive_copy_objective_values[i] );
        }
        free( elitist_archive_copy );
        free( elitist_archive_copy_objective_values );
        free( elitist_archive_copy_constraint_values );
    }

    for( i = 0; i < elitist_archive_capacity; i++ )
    {
        free( elitist_archive[i] );
        free( elitist_archive_objective_values[i] );
        cleanupSolutionMetadata( elitist_archive_solution_metadata[i] );     
    }
    free( elitist_archive );
    free( elitist_archive_objective_values );
    free( elitist_archive_constraint_values );
    free( elitist_archive_solution_metadata );
    free( objective_discretization );
    
    for( i = 0; i < number_of_parameters; i++ )
        free( MI_matrix[i] );

    free( MI_matrix );
}

void loadProblemData()
{
    switch(problem_index)
    {
        case ZEROMAX_ONEMAX: onemaxLoadProblemData(); break;
        case TRAP5: trap5LoadProblemData(); break;
        case LOTZ: lotzLoadProblemData(); break;
        case KNAPSACK: knapsackLoadProblemData(); break;
        case MAXCUT: maxcutLoadProblemData(); break;
        case MAXSAT: maxsatLoadProblemData(); break;
        case MAXCUT_VS_ONEMAX: maxcutVsOnemaxLoadProblem(); break;
        case BESTOFTRAPS: loadBestOfTraps(); break;
        case BESTOFTRAPS_VS_ONEMAX: loadBestOfTrapsVsOneMax(); break;
        case BESTOFTRAPS_VS_MAXCUT: loadBestOfTrapsVsMaxCut(); break;
        case DISCRETIZED_CONTINOUS_PROBLEM: initializeDiscretizedContinuousProblem(); break;
        default: 
            printf("Cannot load problem data!\n");
            exit(1);
    }
}

void initSolutionMetadata(void* &metadata)
{
    switch (problem_index)
    {
        case BESTOFTRAPS: case BESTOFTRAPS_VS_ONEMAX: case BESTOFTRAPS_VS_MAXCUT:
            metadata = initBestOfTrapsMetadata();
            break;
        default:
            // Default: set to null. Most problems do not use the metadata facility.
            metadata = NULL;
    }
    
}

void copySolutionMetadata(void* &metadata_from, void* &metadata_to)
{
    switch(problem_index)
    {
        case BESTOFTRAPS: case BESTOFTRAPS_VS_ONEMAX: case BESTOFTRAPS_VS_MAXCUT:
            copyBestOfTrapsSolutionMetadata(metadata_from, metadata_to);
            break;
        default:
            // By default, metadata should be null, as the default is to not initialize anything.
            // If this assert is triggered, add something to this switch if you are using metadata.
            // Otherwise something went horribly wrong.
            if (metadata_from != NULL || metadata_to != NULL)
            {
                std::cerr << "Got non-null metadata while metadata should be null for problem index.\n";
                std::cerr << "This is a bug that should be fixed: we are likely dealing with uninitialized memory.\n";
                std::cerr << "metadata_from: " << metadata_from << "; metadata_to: " << metadata_to << "\n";
            }

            assert(metadata_from == NULL);
            assert(metadata_to == NULL);
    }
}

// returns true if any headers were written (i.e. do we need to add a comma afterwards?).
bool writeCSVHeaderMetadata(std::ostream &outstream, bool has_preceding_field)
{
    switch(problem_index)
    {
        case BESTOFTRAPS: case BESTOFTRAPS_VS_ONEMAX: case BESTOFTRAPS_VS_MAXCUT:
            return writeCSVHeaderBestOfTrapsMetadata(outstream, has_preceding_field);
        default:
            return false;
    }
}
// returns true if any fields were written (i.e. do we need to add a comma afterwards?).
bool writeCSVMetadata(std::ostream &outstream, void* metadata, bool has_preceding_field)
{
    switch(problem_index)
    {
        case BESTOFTRAPS: case BESTOFTRAPS_VS_ONEMAX: case BESTOFTRAPS_VS_MAXCUT:
            return writeCSVBestOfTrapsMetadata(outstream, metadata, has_preceding_field);
        default:
            return false;
    }
}

void ezilaitiniProblemData()
{
    double **default_front;
    int i, default_front_size;

    switch(problem_index)
    {
        case KNAPSACK: ezilaitiniKnapsackProblemData(); break;
        case MAXCUT: ezilaitiniMaxcutProblemData(); break;
    }

    free(optimization);
    
    default_front = getDefaultFront( &default_front_size );
    if( default_front )
    {
        for( i = 0; i < default_front_size; i++ )
            free( default_front[i] );
        free( default_front );
    }
}

void cleanupSolutionMetadata(void* &metadata)
{
    switch(problem_index)
    {
        case BESTOFTRAPS: case BESTOFTRAPS_VS_ONEMAX: case BESTOFTRAPS_VS_MAXCUT:
            cleanupBestOfTrapsMetadata(metadata);
            break;
        default:
            // By default, metadata should be null, as the default is to not initialize anything.
            // If this assert is triggered: add a case to this switch!
            assert(metadata == NULL);
    }
}

void run( void )
{
    loadProblemData();

    initializeCommonVariables();

    schedule();

    ezilaitiniProblemData();

    ezilaitiniCommonVariables();
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Main -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/
/**
 * The main function:
 * - interpret parameters on the command line
 * - run the algorithm with the interpreted parameters
 */
int main( int argc, char **argv )
{
    interpretCommandLine( argc, argv );

    time_at_start = std::chrono::system_clock::now();
    run();

    return( 0 );
}
