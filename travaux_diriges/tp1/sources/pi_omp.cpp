#include <chrono>
#include <random>
#include <cstdlib>
#include <iostream>
#include <iomanip>

#if defined(_OPENMP)
  #include <omp.h>
#endif

double approximate_pi_omp(unsigned long long nbSamples)
{
    using clock_t = std::chrono::high_resolution_clock;
    unsigned long long inside = 0ULL;

    // Semilla base a partir del tiempo
    unsigned long long baseSeed =
        (unsigned long long)clock_t::now().time_since_epoch().count();

    #pragma omp parallel reduction(+:inside)
    {
        // Un generador por hilo (thread-safe)
        unsigned long long seed = baseSeed;

        #if defined(_OPENMP)
          int tid = omp_get_thread_num();
          seed ^= 0x9e3779b97f4a7c15ULL * (unsigned long long)(tid + 1);
        #else
          seed ^= 0x9e3779b97f4a7c15ULL;
        #endif

        std::mt19937_64 gen(seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        #pragma omp for schedule(static)
        for (unsigned long long s = 0ULL; s < nbSamples; ++s) {
            double x = dist(gen);
            double y = dist(gen);
            if (x*x + y*y <= 1.0) inside += 1ULL;
        }
    }

    return 4.0 * (double)inside / (double)nbSamples;
}

int main(int argc, char* argv[])
{
    unsigned long long nbSamples = 10000000ULL; // 1e7 por defecto
    if (argc > 1) {
        nbSamples = std::strtoull(argv[1], nullptr, 10);
        if (nbSamples == 0ULL) nbSamples = 10000000ULL;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    double pi = approximate_pi_omp(nbSamples);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t1 - t0;

    #if defined(_OPENMP)
      int threads = omp_get_max_threads();
    #else
      int threads = 1;
    #endif

    std::cout << "OpenMP threads: " << threads << "\n";
    std::cout << "Samples: " << nbSamples << "\n";
    std::cout << "Pi ≈ " << std::setprecision(17) << pi << "\n";
    std::cout << "Tiempo (s): " << dt.count() << "\n";
    std::cout << "Muestras/s: " << (double)nbSamples / dt.count() << "\n";

    return 0;
}
