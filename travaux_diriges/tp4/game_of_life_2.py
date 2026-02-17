"""
Le jeu de la vie
################
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

Cette version parallélise le calcul avec décomposition de domaine sur plusieurs processus.
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
       grid = Grille( (10,10), init_pattern=[(2,2),(0,2),(4,2),(2,0),(2,4)], color_life=pg.Color("red"), color_dead=pg.Color("black"))
    """
    def __init__(self, dim, init_pattern=None, color_life=pg.Color("black"), color_dead=pg.Color("white")):
        import random
        self.dimensions = dim
        if init_pattern is not None:
            self.cells = np.zeros(self.dimensions, dtype=np.uint8)
            indices_i = [v[0] for v in init_pattern]
            indices_j = [v[1] for v in init_pattern]
            self.cells[indices_i,indices_j] = 1
        else:
            self.cells = np.random.randint(2, size=dim, dtype=np.uint8)
        self.col_life = color_life
        self.col_dead = color_dead

    def compute_next_iteration(self):
        """
        Calcule la prochaine génération de cellules en suivant les règles du jeu de la vie
        """
        # Remarque 1: on pourrait optimiser en faisant du vectoriel, mais pour plus de clarté, on utilise les boucles
        # Remarque 2: on voit la grille plus comme une matrice qu'une grille géométrique. L'indice (0,0) est donc en bas
        #             à gauche de la grille !
        ny = self.dimensions[0]
        nx = self.dimensions[1]
        next_cells = np.empty(self.dimensions, dtype=np.uint8)
        diff_cells = []
        for i in range(ny):
            i_above = (i+ny-1)%ny
            i_below = (i+1)%ny
            for j in range(nx):
                j_left = (j-1+nx)%nx
                j_right= (j+1)%nx
                voisins_i = [i_above,i_above,i_above, i     , i      , i_below, i_below, i_below]
                voisins_j = [j_left ,j      ,j_right, j_left, j_right, j_left , j      , j_right]
                voisines = np.array(self.cells[voisins_i,voisins_j])
                nb_voisines_vivantes = np.sum(voisines)
                if self.cells[i,j] == 1: # Si la cellule est vivante
                    if (nb_voisines_vivantes < 2) or (nb_voisines_vivantes > 3):
                        next_cells[i,j] = 0 # Cas de sous ou sur population, la cellule meurt
                        diff_cells.append(i*nx+j)
                    else:
                        next_cells[i,j] = 1 # Sinon elle reste vivante
                elif nb_voisines_vivantes == 3: # Cas où cellule morte mais entourée exactement de trois vivantes
                    next_cells[i,j] = 1         # Naissance de la cellule
                    diff_cells.append(i*nx+j)
                else:
                    next_cells[i,j] = 0         # Morte, elle reste morte.
        self.cells = next_cells
        return diff_cells

    def copier_vers_array_partage(self, array_partage, dims):
        array_reshaped = np.frombuffer(array_partage.get_obj(), dtype=np.uint8).reshape(dims)
        array_reshaped[:] = self.cells


class App:
    """
    Fenêtre d'affichage Pygame pour le jeu de la vie.
    IMPORTANT: Affiche TOUTE la grille, indépendamment du nombre de processus de calcul.
    La décomposition de domaine est seulement pour les CALCULS, pas pour le rendu.
    
    geometry : tuple (hauteur_pixels, largeur_pixels) 
    dims : dimensions RÉELLES du patrón (invariant, du dictionnaire)
    """
    def __init__(self, geometry, dims, color_life=pg.Color("black"), color_dead=pg.Color("white")):
        self.dimensions = dims
        ny, nx = dims
        height_px, width_px = geometry
        
        # Taille de celda: resolución / celdas_reales
        # Ejemplo: 800px / 100 celdas = 8px/celda (SIEMPRE, 1 core o 12 cores)
        self.size_x = max(1, width_px // nx)
        self.size_y = max(1, height_px // ny)
        
        print(f"[App] Grilla: {ny}x{nx} | Pantalla: {height_px}x{width_px}px | Celda: {self.size_y}x{self.size_x}px")
        
        if self.size_x > 4 and self.size_y > 4:
            self.draw_color = pg.Color('lightgrey')
        else:
            self.draw_color = None
        
        self.width = nx * self.size_x
        self.height = ny * self.size_y
        self.screen = pg.display.set_mode((self.width, self.height))
        self.col_life = color_life
        self.col_dead = color_dead


    def compute_rectangle(self, i: int, j: int):
        """
        Calcul la géométrie du rectangle correspondant à la cellule (i,j)
        """
        return (self.size_x*j, self.height - self.size_y*(i + 1), self.size_x, self.size_y)

    def compute_color(self, i: int, j: int, cells):
        if cells[i,j] == 0:
            return self.col_dead
        else:
            return self.col_life

    def draw(self, cells):
        [self.screen.fill(self.compute_color(i,j, cells), self.compute_rectangle(i,j)) 
         for i in range(self.dimensions[0]) 
         for j in range(self.dimensions[1])]
        
        if (self.draw_color is not None):
            [pg.draw.line(self.screen, self.draw_color, (0,i*self.size_y), (self.width,i*self.size_y)) 
             for i in range(self.dimensions[0])]
            [pg.draw.line(self.screen, self.draw_color, (j*self.size_x,0), (j*self.size_x,self.height)) 
             for j in range(self.dimensions[1])]
             
        pg.display.update()


def run_calcul_decompose(array_partage, dims, init_pattern, max_iter, evt_fin_affichage, evt_fin_calcul, barrier, num_proc, proc_id):
    """
    Processus de calcul avec décomposition de domaine.
    Synchronisation prioritaire : Initialisation -> Barrière -> Signal -> Affichage initial
    """
    ny, nx = dims
    
    # Décomposition de domaine : répartir les lignes entre processus
    lines_per_proc = ny // num_proc
    i_start = proc_id * lines_per_proc
    i_end = i_start + lines_per_proc if proc_id < num_proc - 1 else ny
    
    # INITIALISATION PRIORITAIRE: Seul proc_id==0 crée et charge le patrón
    if proc_id == 0:
        grille = Grille(dims, init_pattern)
        # Copier l'état initial dans l'array partagé AVANT toute barrière
        grille.copier_vers_array_partage(array_partage, dims)
    else:   
        # Les autres processus créent une grille vide pour le calcul local
        grille = Grille(dims, None)
    
    # BARRIÈRE DE DÉPART : Tous les processus attendent que le patrón soit chargé
    barrier.wait()
    
    # SIGNAL AU MAIN : Seul proc_id==0 signale que le patrén est prêt pour affichage
    if proc_id == 0:
        evt_fin_calcul.set()
    
    for iteration in range(max_iter):
        # ÉTAPE 1 : Proc_id==0 attend que l'affichage soit fini
        if proc_id == 0:
            evt_fin_affichage.wait()
            evt_fin_affichage.clear()
        
        # ÉTAPE 2 : BARRIÈRE - Tous les processus attendent le signal evt_fin_affichage
        barrier.wait()
        
        # ÉTAPE 3 : Copia segura - Chaque processus lit les données fraîches
        cells_shared = np.frombuffer(array_partage.get_obj(), dtype=np.uint8).reshape(dims)
        grille.cells = cells_shared.copy()  # Copia de seguridad pour éviter les conditions de course
        
        # Calculer la génération suivante pour [i_start, i_end)
        # Ghost cells garanties par indices modulo (topologie torique)
        next_cells = np.empty(dims, dtype=np.uint8)
        for i in range(i_start, i_end):
            i_above = (i + ny - 1) % ny
            i_below = (i + 1) % ny
            for j in range(nx):
                j_left = (j - 1 + nx) % nx
                j_right = (j + 1) % nx
                nb_voisines = np.sum([grille.cells[i_above, j_left], grille.cells[i_above, j],
                                      grille.cells[i_above, j_right], grille.cells[i, j_left],
                                      grille.cells[i, j_right], grille.cells[i_below, j_left],
                                      grille.cells[i_below, j], grille.cells[i_below, j_right]])
                
                # Appliquer les règles du jeu de la vie
                if grille.cells[i, j] == 1:
                    if nb_voisines < 2 or nb_voisines > 3:
                        next_cells[i, j] = 0
                    else:
                        next_cells[i, j] = 1
                elif nb_voisines == 3:
                    next_cells[i, j] = 1
                else:
                    next_cells[i, j] = 0
        
        # Écrire le domaine local [i_start, i_end) dans l'array partagé
        cells_shared[i_start:i_end] = next_cells[i_start:i_end]
        
        # ÉTAPE 4 : BARRIÈRE - Attendre que tous les domaines soient écrits
        barrier.wait()
        
        # ÉTAPE 5 : Seul proc_id==0 notifie
        if proc_id == 0:
            # Signal au main qu'une nouvelle génération est prête
            evt_fin_calcul.set()



if __name__ == '__main__':
    dico_patterns = { # Dimension et pattern dans un tuple
        'blinker' : ((5,5),[(2,1),(2,2),(2,3)]),
        'toad'    : ((6,6),[(2,2),(2,3),(2,4),(3,3),(3,4),(3,5)]),
        "acorn"   : ((100,100), [(51,52),(52,54),(53,51),(53,52),(53,55),(53,56),(53,57)]),
        "beacon"  : ((6,6), [(1,3),(1,4),(2,3),(2,4),(3,1),(3,2),(4,1),(4,2)]),
        "boat" : ((5,5),[(1,1),(1,2),(2,1),(2,3),(3,2)]),
        "glider": ((100,90),[(1,1),(2,2),(2,3),(3,1),(3,2)]),
        "glider_gun": ((400,400),[(51,76),(52,74),(52,76),(53,64),(53,65),(53,72),(53,73),(53,86),(53,87),(54,63),(54,67),(54,72),(54,73),(54,86),(54,87),(55,52),(55,53),(55,62),(55,68),(55,72),(55,73),(56,52),(56,53),(56,62),(56,66),(56,68),(56,69),(56,74),(56,76),(57,62),(57,68),(57,76),(58,63),(58,67),(59,64),(59,65)]),
        "space_ship": ((25,25),[(11,13),(11,14),(12,11),(12,12),(12,14),(12,15),(13,11),(13,12),(13,13),(13,14),(14,12),(14,13)]),
        "die_hard" : ((100,100), [(51,57),(52,51),(52,52),(53,52),(53,56),(53,57),(53,58)]),
        "pulsar": ((17,17),[(2,4),(2,5),(2,6),(7,4),(7,5),(7,6),(9,4),(9,5),(9,6),(14,4),(14,5),(14,6),(2,10),(2,11),(2,12),(7,10),(7,11),(7,12),(9,10),(9,11),(9,12),(14,10),(14,11),(14,12),(4,2),(5,2),(6,2),(4,7),(5,7),(6,7),(4,9),(5,9),(6,9),(4,14),(5,14),(6,14),(10,2),(11,2),(12,2),(10,7),(11,7),(12,7),(10,9),(11,9),(12,9),(10,14),(11,14),(12,14)]),
        "floraison" : ((40,40), [(19,18),(19,19),(19,20),(20,17),(20,19),(20,21),(21,18),(21,19),(21,20)]),
        "block_switch_engine" : ((400,400), [(201,202),(201,203),(202,202),(202,203),(211,203),(212,204),(212,202),(214,204),(214,201),(215,201),(215,202),(216,201)]),
        "u" : ((200,200), [(101,101),(102,102),(103,102),(103,101),(104,103),(105,103),(105,102),(105,101),(105,105),(103,105),(102,105),(101,105),(101,104)]),
        "flat" : ((200,400), [(80,200),(81,200),(82,200),(83,200),(84,200),(85,200),(86,200),(87,200), (89,200),(90,200),(91,200),(92,200),(93,200),(97,200),(98,200),(99,200),(106,200),(107,200),(108,200),(109,200),(110,200),(111,200),(112,200),(114,200),(115,200),(116,200),(117,200),(118,200)])
    }
    # Valeurs par défaut
    choice = 'glider'
    nb_proc = 1
    resx = 800
    resy = 800
    
    # Traiter les arguments de ligne de commande avec lógica robusta
    # Ordre: argv[1]=choice, argv[2]=nb_proc, argv[3]=resx, argv[4]=resy
    try:
        if len(sys.argv) > 1:
            choice = sys.argv[1]
        if len(sys.argv) > 2:
            nb_proc = int(sys.argv[2])
        if len(sys.argv) > 3:
            resx = int(sys.argv[3])
        if len(sys.argv) > 4:
            resy = int(sys.argv[4])
    except (ValueError, IndexError) as e:
        print(f"Erreur dans les arguments : {e}. Utilisation des valeurs par défaut.")
        choice = 'glider'
        nb_proc = 1
        resx = 800
        resy = 800
    
    print(f"Pattern initial choisi : {choice}")
    print(f"Resolution écran : {resx}x{resy}")
    print(f"Nombre de processus : {nb_proc}")
    
    try:
        init_pattern = dico_patterns[choice]
    except KeyError:
        print("No such pattern. Available ones are:", dico_patterns.keys())
        exit(1)
    
    # Récupérer les dimensions et le pattern initial
    dims, pattern = init_pattern
    
    # Créer un array partagé pour la grille
    array_partage = mp.Array(ctypes.c_uint8, dims[0] * dims[1])
    
    # Créer les événements de synchronisation
    evt_fin_affichage = mp.Event()
    evt_fin_calcul = mp.Event()
    
    # Créer une barrière pour synchroniser tous les processus de calcul
    barrier = mp.Barrier(nb_proc)
    
    max_iter = 20
    
    # Lancer nb_proc processus de calcul avec décomposition de domaine
    processus_calculs = []
    for proc_id in range(nb_proc):
        p = mp.Process(
            target=run_calcul_decompose,
            args=(array_partage, dims, pattern, max_iter, evt_fin_affichage, evt_fin_calcul, barrier, nb_proc, proc_id)
        )
        p.start()
        processus_calculs.append(p)
    
    # Attendre que les processus aient chargé le patrón initial
    evt_fin_calcul.wait()
    evt_fin_calcul.clear()
    
    # Initialiser pygame et créer la fenêtre
    pg.init()
    appli = App((resx, resy), dims)
    
    mustContinue = True
    count = 0
    stats_calcul = []
    stats_affichage = []
    
    # Vue unique du array partagé
    cells_array = np.frombuffer(array_partage.get_obj(), dtype=np.uint8).reshape(dims)
    
    # Afficher la PREMIÈRE IMAGE (patrón initial) AVANT de lancer les calculs
    t1_aff = time.time()
    appli.draw(cells_array)
    t2_aff = time.time()
    stats_affichage.append(t2_aff - t1_aff)
    
    # Signaler aux processus de calcul qu'ils peuvent commencer (génération 2)
    evt_fin_affichage.set()
    
    # Boucle d'affichage pour les générations suivantes
    while mustContinue and count < max_iter - 1:
        # Mesurer le temps de calcul (depuis affichage jusqu'à calcul terminé)
        t1_calc = time.time()
        
        # Signaler aux processus de démarrer le calcul
        evt_fin_affichage.set()
        
        # Attendre que le calcul soit terminé
        evt_fin_calcul.wait()
        evt_fin_calcul.clear()
        
        t2_calc = time.time()
        stats_calcul.append(t2_calc - t1_calc)
        
        # Mesurer le temps d'affichage
        t1_aff = time.time()
        
        # Afficher la grille mise à jour par les processus
        appli.draw(cells_array)
        
        t2_aff = time.time()
        stats_affichage.append(t2_aff - t1_aff)
        
        count += 1
        print(f"Iteration {count}/{max_iter}", end='\r')
        
        # Gestion des événements
        for event in pg.event.get():
            if event.type == pg.QUIT:
                mustContinue = False
    
    # Libérer ressources graphiques en premier
    pg.quit()
    
    # Fermeture des processus sans blocage
    print("\nFin des itérations. Fermeture des processus...")
    for p in processus_calculs:
        if p.is_alive():
            evt_fin_affichage.set()  # Signal final pour débloquer les processus
            p.terminate()  # Terminer le processus immédiatement
    
    # Statistiques (calculs corrects après les itérations)
    if stats_calcul and len(stats_calcul) > 0:
        avg_calc = sum(stats_calcul) / len(stats_calcul)
        min_calc = min(stats_calcul)
        max_calc = max(stats_calcul)
    else:
        avg_calc = min_calc = max_calc = 0
        
    if stats_affichage and len(stats_affichage) > 0:
        avg_aff = sum(stats_affichage) / len(stats_affichage)
        min_aff = min(stats_affichage)
        max_aff = max(stats_affichage)
    else:
        avg_aff = min_aff = max_aff = 0
        
    avg_total = avg_calc + avg_aff

    with open(f"results_2_({resx}x{resy}_{nb_proc}_proc).txt", "a") as f:
        f.write(f"\nRésultats pour {choice} ({resx}x{resy}) - {nb_proc} processus ---\n")
        f.write(f"Moyenne Calcul : {avg_calc:.6f} s\n")
        f.write(f"Moyenne Affichage : {avg_aff:.6f} s\n")
        f.write(f"Moyenne Totale : {avg_total:.6f} s\n")
        f.write("-" * 40 + "\n")

    print(f"\nCalcul terminé. Résultats sauvegardés dans results_2_({resx}x{resy}_{nb_proc}_proc).txt")
    os._exit(0)  # Forcer fermeture propre de Python sans attendre