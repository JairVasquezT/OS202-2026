from mpi4py import MPI
import numpy as np
from time import time
import os
# IMPORTANTE: Limitar threads de BLAS ANTES de importar numpy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

comm = MPI.COMM_WORLD
nbp = comm.Get_size()
rank = comm.Get_rank()

dim = 120
Nloc = dim // nbp  # Cada proceso toma Nloc filas

# 1. Cada proceso crea solo su bloque de FILAS (Nloc x dim)
A_loc = np.empty((Nloc, dim), dtype=np.double)
for i_loc in range(Nloc):
    i_global = rank * Nloc + i_loc
    for j in range(dim):
        A_loc[i_loc, j] = (i_global + j) % dim + 1.

# 2. Vector u completo en todos los procesos
u = np.array([float(i + 1) for i in range(dim)])

# 3. Producto parcial: (Nloc x dim) * (dim) -> (Nloc)
deb = time()
v_local = A_loc.dot(u)

# 4. Reunir todos los trozos (Allgather)
v_final = np.empty(dim, dtype=np.double)
comm.Allgather(v_local, v_final)
fin = time()

if rank == 0:
    print(f"--- Resultado Matriz-Vector (Filas) ---")
    print(f"Dimension: {dim}x{dim}, Procesos: {nbp}")
    print(f"Primeros 5 elementos de v: {v_final[:5]}")
    print(f"Tiempo de cálculo: {fin-deb:.6f}s")