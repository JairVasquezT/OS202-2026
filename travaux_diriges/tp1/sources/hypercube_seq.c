#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

   
    if (argc != 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <d>\n", argv[0]);
            fprintf(stderr, "Ejemplo: mpirun -np 8 %s 3\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int d = atoi(argv[1]);
    if (d < 0) {
        if (rank == 0) fprintf(stderr, "Error: d debe ser >= 0\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

   
    int expected = 1 << d;
    if (size != expected) {
        if (rank == 0) {
            fprintf(stderr,
                    "Error: para d=%d se requieren %d procesos (2^d), pero se lanzaron %d.\n",
                    d, expected, size);
            fprintf(stderr, "Ejemplo: mpirun -np %d %s %d\n", expected, argv[0], d);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int token = 0;
    if (rank == 0) token = 1;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int step = 1; step < size; ++step) {
        if (rank == step - 1) {
            MPI_Send(&token, 1, MPI_INT, step, 0, MPI_COMM_WORLD);
        } else if (rank == step) {
            MPI_Recv(&token, 1, MPI_INT, step - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double elapsed = t1 - t0;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    printf("rank=%d token=%d\n", rank, token);
    if (rank == 0) {
        printf("Hypercube (Sequentiel) - Temps max [s]: %.6f\n", max_elapsed);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
