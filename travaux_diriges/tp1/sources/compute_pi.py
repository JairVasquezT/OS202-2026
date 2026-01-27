# pi_mpi4py.py
# Approximation de pi par Monte Carlo avec MPI (mpi4py)
# Exécution: mpirun -np 4 python3 pi_mpi4py.py 40000000

import time
import numpy as np
from mpi4py import MPI


def split_samples(total, nbp, rank):
    """Répartition équilibrée: différence max = 1."""
    base = total // nbp
    rem = total % nbp
    return base + (1 if rank < rem else 0)


def main():
    globCom = MPI.COMM_WORLD.Dup()
    nbp = globCom.size
    rank = globCom.rank

    # Nombre total d'échantillons (par défaut)
    total_samples = 40_000_000
    if len(__import__("sys").argv) > 1:
        try:
            total_samples = int(__import__("sys").argv[1])
            if total_samples <= 0:
                total_samples = 40_000_000
        except Exception:
            total_samples = 40_000_000

    local_samples = split_samples(total_samples, nbp, rank)

    # Seed différent par processus (évite corrélation)
    seed = int(time.time_ns() ^ (0x9E3779B97F4A7C15 * (rank + 1))) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)

    # Synchronisation pour mesurer proprement
    globCom.Barrier()
    t0 = time.time()

    # Tirage local (évite de créer trop de temporaires)
    x = rng.random(local_samples) * 2.0 - 1.0
    y = rng.random(local_samples) * 2.0 - 1.0
    inside = np.count_nonzero(x * x + y * y <= 1.0)

    globCom.Barrier()
    t1 = time.time()
    local_time = t1 - t0

    # Réduction des résultats
    total_inside = globCom.reduce(inside, op=MPI.SUM, root=0)
    total_time = globCom.reduce(local_time, op=MPI.MAX, root=0)  # temps parallèle = max

    # Sortie par rang (optionnelle)
    with open(f"Output{rank:03d}.txt", "w") as f:
        f.write(f"rank={rank} nbp={nbp}\n")
        f.write(f"local_samples={local_samples}\n")
        f.write(f"inside={inside}\n")
        f.write(f"local_time_s={local_time:.6f}\n")

    if rank == 0:
        pi = 4.0 * (total_inside / float(total_samples))
        mop_s = (2.0 * total_samples) / total_time / 1e6  # ~2 ops / sample (approx)
        print(f"Pi ≈ {pi:.10f}")
        print(f"Temps (max) = {total_time:.6f} s")
        print(f"Rendement ≈ {mop_s:.1f} Mop/s")
        print(f"Samples = {total_samples} | Processus = {nbp}")


if __name__ == "__main__":
    main()
