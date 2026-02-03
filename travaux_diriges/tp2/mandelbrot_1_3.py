import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from mpi4py import MPI
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm

comm = MPI.COMM_WORLD
nbp = comm.Get_size()
rank = comm.Get_rank()

@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius:  float = 2.0
    def convergence(self, c: complex, smooth=False) -> float:
        z = 0
        for iter in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth: return iter + 1 - log(log(abs(z)))/log(2)
                return iter
        return self.max_iterations

mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1024, 1024
scaleX, scaleY = 3./width, 2.25/height

deb = time()

if rank == 0:
    # --- LÓGICA DEL MAESTRO ---
    full_convergence = np.empty((height, width), dtype=np.double)
    filas_enviadas = 0
    filas_recibidas = 0

    # 1. Enviar una fila inicial a cada esclavo
    for p in range(1, nbp):
        if filas_enviadas < height:
            comm.send(filas_enviadas, dest=p)
            filas_enviadas += 1

    # 2. Recibir resultados y enviar nuevas tareas
    while filas_recibidas < height:
        status = MPI.Status()
        # Recibir (indice_fila, datos_calculados)
        indice_fila, datos_fila = comm.recv(source=MPI.ANY_SOURCE, status=status)
        esclavo_que_envia = status.Get_source()
        
        full_convergence[indice_fila] = datos_fila
        filas_recibidas += 1

        if filas_enviadas < height:
            comm.send(filas_enviadas, dest=esclavo_que_envia)
            filas_enviadas += 1
        else:
            # Enviar señal de terminación (-1)
            comm.send(-1, dest=esclavo_que_envia)

    fin = time()
    print(f"Tiempo total Maestro-Esclavo: {fin-deb:.3f}s")
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(full_convergence)*255))
    image.save("mandelbrot_maestro.png")
    image.show()

else:
    # --- LÓGICA DEL ESCLAVO ---
    while True:
        # Recibir número de fila a calcular
        y = comm.recv(source=0)
        if y == -1: # Si el maestro envía -1, terminamos
            break
        
        # Calcular la fila y
        fila_datos = np.empty(width, dtype=np.double)
        for x in range(width):
            c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
            fila_datos[x] = mandelbrot_set.convergence(c, smooth=True)
        
        # Enviar resultado al maestro: (indice, datos)
        comm.send((y, fila_datos), dest=0)