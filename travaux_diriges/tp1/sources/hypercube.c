#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) fprintf(stderr, "Usage: %s <d>\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int d = atoi(argv[1]);
    int expected = 1 << d;
    if (size != expected) {
        if (rank == 0) fprintf(stderr, "Erreur: pour d=%d, il faut %d processus.\n", d, expected);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int token = (rank == 0) ? 1 : 0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Algorithme de diffusion en Hypercube
    for (int k = 0; k < d; ++k) {
        int neighbor = rank ^ (1 << k);
        if (rank < (1 << k)) {
            MPI_Send(&token, 1, MPI_INT, neighbor, k, MPI_COMM_WORLD);
        } else if (rank < (1 << (k + 1))) {
            MPI_Recv(&token, 1, MPI_INT, neighbor, k, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double elapsed = t1 - t0;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Sauvegarde des résultats
    char fileName[20];
    sprintf(fileName, "Output_Hyper_%02d.txt", rank);
    FILE* f = fopen(fileName, "w");
    fprintf(f, "Rang: %d, Jeton reçu: %d, Temps: %.6f s\n", rank, token, elapsed);
    fclose(f);

    if (rank == 0) {
        printf("Hypercube (Parallele) - Temps max [s]: %.6f\n", max_elapsed);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}   