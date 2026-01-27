#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int token = 0;
    int tag = 0;
    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;

    if (rank == 0) {
        token = 1;
        MPI_Send(&token, 1, MPI_INT, next, tag, MPI_COMM_WORLD);
        MPI_Recv(&token, 1, MPI_INT, prev, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Token final en rank 0: %d\n", token);
    } else {
        MPI_Recv(&token, 1, MPI_INT, prev, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        token += 1;
        MPI_Send(&token, 1, MPI_INT, next, tag, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
