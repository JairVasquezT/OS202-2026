"""
Le jeu de la vie - Version Parallèle Vectorisée avec Décomposition de Domaine
###############################################################################
Le jeu de la vie est un automate cellulaire inventé par Conway se basant normalement sur une grille infinie
de cellules en deux dimensions. Ces cellules peuvent prendre deux états :
    - un état vivant
    - un état mort
A l'initialisation, certaines cellules sont vivantes, d'autres mortes.
Le principe du jeu est alors d'itérer de telle sorte qu'à chaque itération, une cellule va devoir interagir avec
les huit cellules voisines (gauche, droite, bas, haut et les quatre en diagonales.) L'interaction se fait selon les
règles suivantes pour calculer l'irération suivante :
    - Une cellule vivante avec moins de deux cellules voisines vivantes meurt ( sous-population )
    - Une cellule vivante avec deux ou trois cellules voisines vivantes reste vivante
    - Une cellule vivante avec plus de trois cellules voisines vivantes meurt ( sur-population )
    - Une cellule morte avec exactement trois cellules voisines vivantes devient vivante ( reproduction )

Pour ce projet, on change légèrement les règles en transformant la grille infinie en un tore contenant un
nombre fini de cellules. Les cellules les plus à gauche ont pour voisines les cellules les plus à droite
et inversement, et de même les cellules les plus en haut ont pour voisines les cellules les plus en bas
et inversement.

Cette version parallelise le calcul avec décomposition de domaine et utilise la vectorisation NumPy pour optimiser.
"""
import pygame as pg
import numpy as np
import multiprocessing as mp
import ctypes
import time
import sys
import os


class Grille:
    """
    Grille torique décrivant l'automate cellulaire.
    En entrée lors de la création de la grille :
        - dimensions est un tuple contenant le nombre de cellules dans les deux directions (nombre lignes, nombre colonnes)
        - init_pattern est une liste de cellules initialement vivantes sur cette grille (les autres sont considérées comme mortes)
        - color_life est la couleur dans laquelle on affiche une cellule vivante
        - color_dead est la couleur dans laquelle on affiche une cellule morte
    Si aucun pattern n'est donné, on tire au hasard quels sont les cellules vivantes et les cellules mortes
    Exemple :
       grid = Grille((10,10), init_pattern=[(2,2),(0,2),(4,2),(2,0),(2,4)], color_life=pg.Color("red"), color_dead=pg.Color("black"))
    """
    def __init__(self, dim, init_pattern=None, color_life=pg.Color("black"), color_dead=pg.Color("white")):
        import random
        self.dimensions = dim
        if init_pattern is not None:
            self.cells = np.zeros(self.dimensions, dtype=np.uint8)
            indices_i = [v[0] for v in init_pattern]
            indices_j = [v[1] for v in init_pattern]
            self.cells[indices_i, indices_j] = 1
        else:
            self.cells = np.random.randint(2, size=dim, dtype=np.uint8)
        self.col_life = color_life
        self.col_dead = color_dead

    def compute_next_iteration_vectorized(self, i_start, i_end):
        """
        Calcule la prochaine génération de cellules en suivant les règles du jeu de la vie
        """
        ny, nx = self.dimensions
        
        # Calcular vecinos usando np.roll (topología toroidal automática)
        N = np.roll(self.cells, 1, axis=0)    # Una fila arriba
        S = np.roll(self.cells, -1, axis=0)   # Una fila abajo
        E = np.roll(self.cells, -1, axis=1)   # Una columna a la derecha
        W = np.roll(self.cells, 1, axis=1)    # Una columna a la izquierda
        NE = np.roll(np.roll(self.cells, 1, axis=0), -1, axis=1)   # Diagonal NE
        NW = np.roll(np.roll(self.cells, 1, axis=0), 1, axis=1)    # Diagonal NW
        SE = np.roll(np.roll(self.cells, -1, axis=0), -1, axis=1)  # Diagonal SE
        SW = np.roll(np.roll(self.cells, -1, axis=0), 1, axis=1)   # Diagonal SW
        
        # Suma de voisines (vecinos)
        nb_voisines = N + S + E + W + NE + NW + SE + SW
        
        # Aplicar reglas de Conway de forma vectorizada
        # La célula vive si: está viva con 2 o 3 voisines, O está muerta con 3 voisines
        next_cells = np.zeros(self.dimensions, dtype=np.uint8)
        next_cells[(self.cells == 1) & ((nb_voisines == 2) | (nb_voisines == 3))] = 1  # Supervivencia
        next_cells[(self.cells == 0) & (nb_voisines == 3)] = 1  # Natalidad
        
        return next_cells[i_start:i_end]

    def copier_vers_array_partage(self, array_partage, dims):
        """Copie l'état de la grille vers l'array partagé entre processus"""
        array_reshaped = np.frombuffer(array_partage.get_obj(), dtype=np.uint8).reshape(dims)
        array_reshaped[:] = self.cells


class App:
    """
    Cette classe décrit la fenêtre affichant la grille à l'écran.
    Affiche TOUTE la grille indépendamment du nombre de processus.
    La décomposition de domaine est seulement pour les CALCULS, pas pour le rendu.
    
        - geometry est un tuple de deux entiers donnant (hauteur_px, largeur_px)
        - dims est la grille décrivant l'automate cellulaire
    """
    def __init__(self, geometry, dims, color_life=pg.Color("black"), color_dead=pg.Color("white")):
        self.dimensions = dims  # (ny, nx) - invariant, never changes with nb_proc
        height_px, width_px = geometry
        
        # Calcul de la taille d'une cellule par rapport à la taille de la fenêtre et de la grille à afficher :
        self.size_x = width_px // self.dimensions[1]
        self.size_y = height_px // self.dimensions[0]
        
        if self.size_x > 4 and self.size_y > 4:
            self.draw_color = pg.Color('lightgrey')
        else:
            self.draw_color = None
        
        # Ajustement de la taille de la fenêtre pour bien fitter la dimension de la grille
        self.width = self.dimensions[1] * self.size_x
        self.height = self.dimensions[0] * self.size_y
        # Création de la fenêtre
        self.screen = pg.display.set_mode((self.width, self.height))
        self.col_life = color_life
        self.col_dead = color_dead

    def compute_rectangle(self, i: int, j: int):
        """
        Calcul la géométrie du rectangle correspondant à la cellule (i,j)
        """
        return (self.size_x * j, self.height - self.size_y * (i + 1), self.size_x, self.size_y)

    def compute_color(self, i: int, j: int, cells):
        if cells[i, j] == 0:
            return self.col_dead
        else:
            return self.col_life

    def draw(self, cells):
        """
        Affiche TOUTE la grille (dimensions originales du patrón).
        La décomposition de domaine n'affecte PAS le rendu.
        """
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                color = self.compute_color(i, j, cells)
                rect = self.compute_rectangle(i, j)
                self.screen.fill(color, rect)
        
        if self.draw_color is not None:
            for i in range(self.dimensions[0]):
                y = self.height - self.size_y * (i + 1)
                pg.draw.line(self.screen, self.draw_color, (0, y), (self.width, y))
            for j in range(self.dimensions[1]):
                x = self.size_x * j
                pg.draw.line(self.screen, self.draw_color, (x, 0), (x, self.height))
        
        pg.display.update()


def run_calcul_decompose_vect(array_partage, dims, init_pattern, max_iter,
                               evt_fin_affichage, evt_fin_calcul, barrier, num_proc, proc_id):
    """
    Función ejecutada por cada proceso hijo.
    Realiza cálculos con descomposición de dominio y vectorización NumPy.
    
    Sincronización: Barrera + Eventos para coordinar cálculo y visualización.
    Vectorización: Usa np.roll para calcular voisins de forma eficiente.
    """
    ny, nx = dims
    
    # Descomposición de dominio: repartir líneas entre procesos
    lines_per_proc = ny // num_proc
    i_start = proc_id * lines_per_proc
    i_end = i_start + lines_per_proc if proc_id < num_proc - 1 else ny
    
    # INICIALIZACIÓN: Solo proc_id==0 crea y carga el patrón
    if proc_id == 0:
        grille = Grille(dims, init_pattern)
        grille.copier_vers_array_partage(array_partage, dims)
    else:
        grille = Grille(dims, None)
    
    # Barrera: Todos esperan a que el patrón esté cargado
    barrier.wait()
    
    # Señal al main: proc_id==0 indica que el patrón está listo
    if proc_id == 0:
        evt_fin_calcul.set()
    
    # Loop de iteraciones
    for iteration in range(max_iter):
        # Paso 1: proc_id==0 espera que el display esté listo
        if proc_id == 0:
            evt_fin_affichage.wait()
            evt_fin_affichage.clear()
        
        # Paso 2: Barrera - todos esperan el signal evt_fin_affichage
        barrier.wait()
        
        # Paso 3: Copia segura - cada proceso lee datos frescos
        cells_shared = np.frombuffer(array_partage.get_obj(), dtype=np.uint8).reshape(dims)
        # Hacer una copia completa para que la vectorización (np.roll) tenga la grilla entera
        grille.cells = cells_shared.copy()

        # Paso 4: Barrera intermedia - todos han leído una copia completa antes de calcular
        barrier.wait()

        # Paso 5: Calcular la próxima generación usando vectorización NumPy (solo la banda local)
        next_cells_band = grille.compute_next_iteration_vectorized(i_start, i_end)

        # Paso 6: Escribir dominio local [i_start, i_end) al array compartido
        cells_shared[i_start:i_end] = next_cells_band

        # Paso 7: Barrera final - esperar a que todos terminen de escribir
        barrier.wait()
        
        # Paso 7: proc_id==0 notifica que nueva generación está lista
        if proc_id == 0:
            evt_fin_calcul.set()


if __name__ == '__main__':
    dico_patterns = {  # Dimension et pattern dans un tuple
        'blinker': ((5, 5), [(2, 1), (2, 2), (2, 3)]),
        'toad': ((6, 6), [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (3, 5)]),
        "acorn": ((100, 100), [(51, 52), (52, 54), (53, 51), (53, 52), (53, 55), (53, 56), (53, 57)]),
        "beacon": ((6, 6), [(1, 3), (1, 4), (2, 3), (2, 4), (3, 1), (3, 2), (4, 1), (4, 2)]),
        "boat": ((5, 5), [(1, 1), (1, 2), (2, 1), (2, 3), (3, 2)]),
        "glider": ((100, 90), [(1, 1), (2, 2), (2, 3), (3, 1), (3, 2)]),
        "glider_gun": ((400, 400), [(51, 76), (52, 74), (52, 76), (53, 64), (53, 65), (53, 72), (53, 73), (53, 86), (53, 87),
                                    (54, 63), (54, 67), (54, 72), (54, 73), (54, 86), (54, 87), (55, 52), (55, 53), (55, 62),
                                    (55, 68), (55, 72), (55, 73), (56, 52), (56, 53), (56, 62), (56, 66), (56, 68), (56, 69),
                                    (56, 74), (56, 76), (57, 62), (57, 68), (57, 76), (58, 63), (58, 67), (59, 64), (59, 65)]),
        "space_ship": ((25, 25), [(11, 13), (11, 14), (12, 11), (12, 12), (12, 14), (12, 15), (13, 11), (13, 12), (13, 13), (13, 14), (14, 12), (14, 13)]),
        "die_hard": ((100, 100), [(51, 57), (52, 51), (52, 52), (53, 52), (53, 56), (53, 57), (53, 58)]),
        "pulsar": ((17, 17), [(2, 4), (2, 5), (2, 6), (7, 4), (7, 5), (7, 6), (9, 4), (9, 5), (9, 6), (14, 4), (14, 5), (14, 6),
                              (2, 10), (2, 11), (2, 12), (7, 10), (7, 11), (7, 12), (9, 10), (9, 11), (9, 12), (14, 10), (14, 11), (14, 12),
                              (4, 2), (5, 2), (6, 2), (4, 7), (5, 7), (6, 7), (4, 9), (5, 9), (6, 9), (4, 14), (5, 14), (6, 14),
                              (10, 2), (11, 2), (12, 2), (10, 7), (11, 7), (12, 7), (10, 9), (11, 9), (12, 9), (10, 14), (11, 14), (12, 14)]),
        "floraison": ((40, 40), [(19, 18), (19, 19), (19, 20), (20, 17), (20, 19), (20, 21), (21, 18), (21, 19), (21, 20)]),
        "block_switch_engine": ((400, 400), [(201, 202), (201, 203), (202, 202), (202, 203), (211, 203), (212, 204), (212, 202),
                                             (214, 204), (214, 201), (215, 201), (215, 202), (216, 201)]),
        "u": ((200, 200), [(101, 101), (102, 102), (103, 102), (103, 101), (104, 103), (105, 103), (105, 102), (105, 101), (105, 105),
                           (103, 105), (102, 105), (101, 105), (101, 104)]),
        "flat": ((200, 400), [(80, 200), (81, 200), (82, 200), (83, 200), (84, 200), (85, 200), (86, 200), (87, 200), (89, 200),
                              (90, 200), (91, 200), (92, 200), (93, 200), (97, 200), (98, 200), (99, 200), (106, 200), (107, 200),
                              (108, 200), (109, 200), (110, 200), (111, 200), (112, 200), (114, 200), (115, 200), (116, 200), (117, 200), (118, 200)])
    }
    
    choice = 'glider'
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    resx = 800
    resy = 800
    if len(sys.argv) > 3:
        resx = int(sys.argv[2])
        resy = int(sys.argv[3])
    nb_proc = 1
    if len(sys.argv) > 4:
        nb_proc = int(sys.argv[4])
    
    print(f"Pattern initial choisi : {choice}")
    print(f"Resolution ecran : {resx},{resy}")
    print(f"Nombre de processus : {nb_proc}")
    
    try:
        init_pattern = dico_patterns[choice]
    except KeyError:
        print("No such pattern. Available ones are:", list(dico_patterns.keys()))
        exit(1)
    
    dims, pattern = init_pattern
    
    # Créer les primitives de multiprocessing
    array_partage = mp.Array(ctypes.c_uint8, dims[0] * dims[1])
    evt_fin_affichage = mp.Event()
    evt_fin_calcul = mp.Event()
    barrier = mp.Barrier(nb_proc)
    
    max_iter = 20
    
    # Lancer les processus de calcul
    processus_calculs = []
    for proc_id in range(nb_proc):
        p = mp.Process(
            target=run_calcul_decompose_vect,
            args=(array_partage, dims, pattern, max_iter, evt_fin_affichage, evt_fin_calcul, barrier, nb_proc, proc_id)
        )
        p.start()
        processus_calculs.append(p)
    
    # Attendre que le patrón initial soit chargé
    evt_fin_calcul.wait()
    evt_fin_calcul.clear()
    
    # Initialiser pygame
    pg.init()
    appli = App((resx, resy), dims)
    
    count = 0
    mustContinue = True
    stats_calcul = []
    stats_affichage = []
    
    cells_array = np.frombuffer(array_partage.get_obj(), dtype=np.uint8).reshape(dims)
    
    # Afficher l'image initiale
    t1 = time.time()
    appli.draw(cells_array)
    t3 = time.time()
    stats_affichage.append(t3 - t1)
    
    # Signaler le démarrage des calculs
    evt_fin_affichage.set()
    
    # Boucle principale
    while mustContinue and count < max_iter - 1:
        # Comprobar eventos primero para no quedarse bloqueado esperando al cálculo
        for event in pg.event.get():
            if event.type == pg.QUIT:
                mustContinue = False
                # Señal final para desbloquear a los procesos que estén esperando
                evt_fin_affichage.set()
                break
        if not mustContinue:
            break

        t1 = time.time()
        evt_fin_calcul.wait()
        evt_fin_calcul.clear()
        t2 = time.time()
        stats_calcul.append(t2 - t1)

        appli.draw(cells_array)
        t3 = time.time()
        stats_affichage.append(t3 - t2)
        count += 1
        print(f"Iteration {count}/{max_iter}", end='\r')
        evt_fin_affichage.set()
    
    # Libérer ressources graphiques en premier
    pg.quit()
    
    # Fermeture des processus sans blocage
    print("\nFin des itérations. Fermeture des processus...")
    for p in processus_calculs:
        if p.is_alive():
            evt_fin_affichage.set()  # Signal final pour débloquer les processus
            p.terminate()  # Terminer le processus immédiatement
    
    # Statistiques
    avg_calc = sum(stats_calcul) / len(stats_calcul) if stats_calcul else 0
    avg_aff = sum(stats_affichage) / len(stats_affichage) if stats_affichage else 0
    avg_total = avg_calc + avg_aff

    with open(f"results_paralelo_vect_({resx}x{resy}_{nb_proc}_proc).txt", "a") as f:
        f.write(f"\nRésultats pour {choice} ({resx}x{resy}) - {nb_proc} processus --- (Vectorisé + Décomposition domaine)\n")
        f.write(f"Moyenne Calcul : {avg_calc:.6f} s\n")
        f.write(f"Moyenne Affichage : {avg_aff:.6f} s\n")
        f.write(f"Moyenne Totale : {avg_total:.6f} s\n")
        f.write("-" * 40 + "\n")

    print(f"\nCalcul terminé. Résultats sauvegardés dans results_paralelo_vect_({resx}x{resy}_{nb_proc}_proc).txt")
    os._exit(0)  # Forcer fermeture propre de Python sans attendre
