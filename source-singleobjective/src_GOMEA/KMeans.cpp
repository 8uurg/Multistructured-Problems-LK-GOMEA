#include <KMeans.hpp>
#include <queue>

typedef void (*initial_centroids)(
    const size_t num_centroids,
    vector<Individual *> &population,
    vector<vector<double>> *centroids,
    mt19937 rng);

typedef bool (*reassign_clusters)(
    vector<Individual *> &population,
    vector<vector<double>> &centroids,
    vector<size_t> *clusters);

void get_initial_centroids_random_sample(
    const size_t num_centroids,
    vector<Individual *> &population,
    vector<vector<double>> *centroids,
    mt19937 rng)
{
    vector<Individual *> sampled;
    centroids->resize(num_centroids);
    sample(population.begin(), population.end(), sampled.begin(), num_centroids, rng);

    for (size_t i = 0; i < num_centroids; ++i)
    {
        auto from = sampled[i]->genotype;
        for (size_t j = 0; j < from.size(); ++j)
        {
            // NOTE: We assume that the genotype is continuous here (by converting to a double)...
            // Not neccesarily the case.
            // Alternative requires kernelized k-means?
            (*centroids)[i][j] = (double)from[j];
        }
    }
}

// For Standard KMeans
bool reassign_clusters_nearest(
    vector<Individual *> &population,
    vector<vector<double>> &centroids,
    vector<size_t> *clusters)
{
    const size_t pop_size = population.size();
    const size_t num_centroids = centroids.size();
    const size_t dimensionality = centroids[0].size();

    bool changed = false;

    for (size_t i = 0; i < pop_size; ++i)
    {
        size_t nearest = 0;
        double distance_nearest = INFINITY;
        for (size_t j = 0; j < num_centroids; ++j)
        {
            // Hardcoded euclidean distance...
            double distance = 0.0;
            for (size_t d = 0; d < dimensionality; ++d)
            {
                double diff = (centroids[j][d] - ((double)population[i]->genotype[d]));
                double diffsq = diff * diff;
                distance += diffsq;
            }
            distance = sqrt(distance);

            // Find nearest neighbor
            if (distance < distance_nearest)
            {
                nearest = j;
                distance_nearest = distance;
            }
        }
        if ((*clusters)[i] != nearest)
        {
            changed = true;
        }
        (*clusters)[i] = nearest;
    }

    return changed;
}

// For Balanced KMeans via
// Malinen, Mikko I., and Pasi Fränti. 2014.
// ‘Balanced K-Means for Clustering’.
// In Structural, Syntactic, and Statistical Pattern Recognition, 32–41.
// Lecture Notes in Computer Science. Berlin, Heidelberg: Springer.
// https://doi.org/10.1007/978-3-662-44415-3_4.
bool reassign_clusters_smallest_distance_equal_assignment(
    vector<Individual *> &population,
    vector<vector<double>> &centroids,
    vector<size_t> *clusters)
{
    const size_t pop_size = population.size();
    const size_t num_centroids = centroids.size();
    const size_t dimensionality = centroids[0].size();

    // Compute and cache all distances.
    vector<double> distances;
    distances.resize(pop_size * num_centroids);
    for (size_t i = 0; i < pop_size; ++i)
    {
        for (size_t j = 0; j < num_centroids; ++j)
        {
            // Hardcoded euclidean distance...
            double distance = 0.0;
            for (size_t d = 0; d < dimensionality; ++d)
            {
                double diff = (centroids[j][d] - ((double)population[i]->genotype[d]));
                double diffsq = diff * diff;
                distance += diffsq;
            }
            distance = sqrt(distance);

            distances[i * pop_size + j] = distance;
        }
    }
    // Hungarian Algorithm / Munkres
    // TODO...

}

void recompute_centroids(
    vector<Individual *> &population,
    vector<size_t> &clusters,
    vector<vector<double>> *centroids)
{
    const size_t pop_size = population.size();
    const size_t num_centroids = centroids->size();
    const size_t dimensionality = (*centroids)[0].size();

    // Get a vector of weights.
    vector<size_t> w;
    w.resize(clusters.size());
    fill(w.begin(), w.end(), 0);

    for (size_t i = 0; i < pop_size; ++i)
    {
        const size_t cluster = clusters[i];
        assert(cluster < num_centroids);
        for (size_t d = 0; d < dimensionality; ++d)
        {
            (*centroids)[cluster][d] = (w[cluster] * (*centroids)[cluster][d] +
                                        (double)population[i]->genotype[d]) /
                                       (w[cluster] + 1);
        }
        w[cluster] += 1;
    }
}

template <
    initial_centroids ic,
    reassign_clusters rc>
void kmeans_general(
    size_t num_clusters,
    vector<Individual *> &population,
    vector<size_t> *clusters,
    vector<vector<double>> *centroids,
    mt19937 rng)
{
    (*ic)(num_clusters, population, centroids, rng);
    (*rc)(population, *centroids, clusters);
    recompute_centroids(population, *clusters, centroids);

    while (true)
    {
        if ((*rc)(population, *centroids, clusters))
        {
            recompute_centroids(population, *clusters, centroids);
        }
        else
        {
            break;
        }
    }
}

void kmeans(
    size_t num_clusters,
    vector<Individual *> &population,
    vector<size_t> *clusters,
    vector<vector<double>> *centroids,
    mt19937 rng)
{
    kmeans_general<
        &get_initial_centroids_random_sample,
        &reassign_clusters_nearest
    >
    (num_clusters, population, clusters, centroids, rng);
}

void balanced_kmeans(
    size_t num_clusters,
    vector<Individual *> &population,
    vector<size_t> *clusters,
    vector<vector<double>> *centroids,
    mt19937 rng)
{
    kmeans_general<
        &get_initial_centroids_random_sample,
        &reassign_clusters_smallest_distance_equal_assignment
    >
    (num_clusters, population, clusters, centroids, rng);
}