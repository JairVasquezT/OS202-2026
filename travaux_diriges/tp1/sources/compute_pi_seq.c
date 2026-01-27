#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

static inline uint32_t xorshift32(uint32_t *state) {
  uint32_t x = *state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;
  return x;
}

int main(int argc, char **argv) {
  unsigned long long nb_samples = 40000000ULL;
  if (argc > 1) {
    char *end = NULL;
    unsigned long long val = strtoull(argv[1], &end, 10);
    if (end && *end == '\0' && val > 0)
      nb_samples = val;
  }
  int threads = 0;
  if (argc > 2) {
    char *end = NULL;
    long val = strtol(argv[2], &end, 10);
    if (end && *end == '\0' && val > 0)
      threads = (int)val;
  }

  struct timespec beg;
  struct timespec end;
  clock_gettime(CLOCK_MONOTONIC, &beg);

  unsigned long long inside = 0;
#if defined(_OPENMP)
  if (threads > 0)
    omp_set_num_threads(threads);
  int used_threads = 1;
  #pragma omp parallel
  {
    #pragma omp single
    used_threads = omp_get_num_threads();

    uint32_t seed = (uint32_t)time(NULL)
                    ^ (uint32_t)(omp_get_thread_num() * 0x9e3779b1u);
    if (seed == 0)
      seed = 1u;

    #pragma omp for reduction(+:inside) schedule(static)
    for (unsigned long long i = 0; i < nb_samples; ++i) {
      double rx = (double)xorshift32(&seed) / (4294967296.0);
      double ry = (double)xorshift32(&seed) / (4294967296.0);
      double x = 2.0 * rx - 1.0;
      double y = 2.0 * ry - 1.0;
      if (x * x + y * y < 1.0)
        ++inside;
    }
  }
  printf("Threads: %d\n", used_threads);
#else
  (void)threads;
  uint32_t seed = (uint32_t)time(NULL);
  if (seed == 0)
    seed = 1u;
  for (unsigned long long i = 0; i < nb_samples; ++i) {
    double rx = (double)xorshift32(&seed) / (4294967296.0);
    double ry = (double)xorshift32(&seed) / (4294967296.0);
    double x = 2.0 * rx - 1.0;
    double y = 2.0 * ry - 1.0;
    if (x * x + y * y < 1.0)
      ++inside;
  }
#endif

  clock_gettime(CLOCK_MONOTONIC, &end);
  double elapsed = (double)(end.tv_sec - beg.tv_sec)
                   + (double)(end.tv_nsec - beg.tv_nsec) / 1e9;

  double approx_pi = 4.0 * (double)inside / (double)nb_samples;
  printf("Temps pour calculer pi : %.6f secondes\n", elapsed);
  printf("Pi vaut environ %.10f\n", approx_pi);
  return 0;
}
