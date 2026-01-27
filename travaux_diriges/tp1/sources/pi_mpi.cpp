#include <mpi.h>
#include <chrono>
#include <random>
#include <cstdlib>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>

struct LocalResult {
    unsigned long long samples = 0ULL;
    unsigned long long dartsInCircle = 0ULL;
};

// Simulation de Monte-Carlo pour approximer Pi
static LocalResult approximate_pi_counts(unsigned long long samples, unsigned long long seed)
{
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    LocalResult r;
    r.samples = samples;

    for (unsigned long long s = 0ULL; s < samples; ++s) {
        double x = dist(gen);
        double y = dist(gen);
        if (x*x + y*y <= 1.0) r.dartsInCircle++;
    }
    return r;
}

int main(int argc, char* argv[])
{
    // Initialisation de l'environnement MPI
    MPI_Init(&argc, &argv);

    MPI_Comm globComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &globComm);

    int nbp = 0, rank = 0;
    MPI_Comm_size(globComm, &nbp);
    MPI_Comm_rank(globComm, &rank);

    // Valeur par défaut : 100 millions d'échantillons
    unsigned long long totalSamples = 100000000ULL; 
    if (argc > 1) {
        totalSamples = std::strtoull(argv[1], nullptr, 10);
        if (totalSamples == 0ULL) totalSamples = 100000000ULL;
    }

    // Répartition équitable de la charge de calcul
    unsigned long long base = totalSamples / (unsigned long long)nbp;
    unsigned long long rem  = totalSamples % (unsigned long long)nbp;
    unsigned long long localSamples = base + ((unsigned long long)rank < rem ? 1ULL : 0ULL);

    // Initialisation de la graine aléatoire unique par rang
    using clock_t = std::chrono::high_resolution_clock;
    unsigned long long t = (unsigned long long)clock_t::now().time_since_epoch().count();
    unsigned long long seed = t ^ (0x9e3779b97f4a7c15ULL * (unsigned long long)(rank + 1));

    // Synchronisation avant de démarrer le chronomètre
    MPI_Barrier(globComm);
    double t0 = MPI_Wtime();

    LocalResult local = approximate_pi_counts(localSamples, seed);

    MPI_Barrier(globComm);
    double t1 = MPI_Wtime();
    double localTime = t1 - t0;

    // Réduction des résultats locaux vers le processus 0
    unsigned long long globalSamples = 0ULL;
    unsigned long long globalDarts   = 0ULL;

    MPI_Reduce(&local.samples, &globalSamples, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, globComm);
    MPI_Reduce(&local.dartsInCircle, &globalDarts, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, globComm);

    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, globComm);

    // Calcul local de Pi pour le fichier de sortie
    double localPi = 4.0 * (double)local.dartsInCircle / (double)local.samples;

    // Sauvegarde des résultats individuels dans des fichiers
    std::stringstream fileName;
    fileName << "Output" << std::setfill('0') << std::setw(5) << rank << ".txt";
    std::ofstream output(fileName.str().c_str());

    output << "Rang=" << rank << " nbp=" << nbp << "\n";
    output << "Echantillons_locaux=" << local.samples << "\n";
    output << "Points_dans_le_cercle=" << local.dartsInCircle << "\n";
    output << "Pi_local=" << std::setprecision(17) << localPi << "\n";
    output << "Temps_local(s)=" << std::setprecision(6) << localTime << "\n";
    output.close();

    // Affichage des résultats globaux par le processus 0
    if (rank == 0) {
        double pi = 4.0 * (double)globalDarts / (double)globalSamples;
        // 2 opérations par échantillon (x*x + y*y)
        double mflops = (2.0 * (double)globalSamples) / maxTime / 1e6; 
        
        std::cout << "Pi global \u2248 " << std::setprecision(17) << pi << "\n";
        std::cout << "Nombre total d'echantillons : " << globalSamples << "\n";
        std::cout << "Temps d'execution (max entre rangs) [s] : " << std::setprecision(6) << maxTime << "\n";
        std::cout << "Performance estimee [Mop/s] : " << std::setprecision(3) << mflops << "\n";
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
