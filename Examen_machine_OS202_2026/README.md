# Simulation N-Body Distribuée avec Parallélisme Hybride

## Introduction

Simulation de dynamique stellaire optimisée par des grilles spatiales et du parallélisme hybride (Numba + MPI). Le projet combine le parallélisme au niveau des threads (Numba JIT avec prange) et au niveau des processus (MPI) pour accélérer le calcul des forces gravitationnelles dans les systèmes à N corps. Il utilise une grille cartésienne 3D avec indexation de Morse pour réduire la complexité de recherche des forces de O(N²) à O(N).

**Version principale:** `nbodies_grid_numba_2_para_v2.py` (Point 4 - MPI Distribuée avec domain decomposition)

---

## Tableau des Résultats de Performance

| Étape du Projet | Configuration | Temps Update Moyen | Speedup (vs 1.3) |
| :--- | :--- | :--- | :--- |
| 1.3 Séquentiel | 1 Noyau | ~1400 ms | 1.0x |
| 2.0 Numba Parallel | 4 Threads | ~380 ms | 3.68x |
| 3.0 MPI Basique | 2 Proc | 126 ms | 11.1x |
| 3.0 MPI Basique | 4 Proc | 245 ms | 5.7x |
| 4.0 MPI Distribuée | 4 Proc (Meilleur) | 111 ms | 12.6x |

Le Point 3 avec 4 processus montre une dégradation par rapport à 2 processus en raison de l'augmentation du surcoût de communication MPI (Gather + Broadcast). Le Point 4 optimise la communication par domain decomposition spatiale, réalisant une meilleure scalabilité.

---

## Analyse du Point 5: Théorie du Déséquilibre de Charge

### Déséquilibre de Charge

La densité de corps dans une galaxie n'est pas uniforme. Le centre (près du trou noir) présente une haute densité, tandis que la périphérie est dispersée. Avec une domain decomposition uniforme dans la dimension X :

### Déséquilibre de Charge

La densité de corps dans une galaxie n'est pas uniforme. Le centre (près du trou noir) présente une haute densité, tandis que la périphérie est dispersée. Avec une domain decomposition uniforme dans la dimension X :

- Processus gérant des cellules centrales : 1000+ corps par cellule
- Processus gérant des cellules périphériques : 10-50 corps par cellule
- Résultat : La synchronisation MPI attend le processus le plus lent (jusqu'à 5x de différence)

Ce phénomène explique pourquoi MPI basique avec 4 processus est plus lent qu'avec 2 processus : le surcoût de synchronisation augmente linéairement tandis que le gain computationnel est limité par le déséquilibre.

### Distribution Intelligente : Courbes de Remplissage d'Espace

Pour répartir la charge des étoiles de manière équitable indépendamment du volume géométrique, on peut utiliser :

1. Grilles Adaptatives : Subdiviser dynamiquement les cellules denses en sous-cellules plus fines
2. Courbes de Remplissage (Z-order ou Hilbert) : Mapper l'espace 3D à une ligne 1D en préservant la localité. Diviser cette ligne en segments égaux (par nombre de corps) assigne à chaque processus approximativement la même charge.

**Avantage:** Localité des données améliorée + équilibre automatique par densité

---

## Instructions d'Utilisation

### Activation de l'Environnement

```bash
cd /home/jair/proyectos/OS202-2026/Examen_machine_OS202_2026
source mi_examen_env/bin/activate
```

### Version Séquentielle (Point 1.3)

```bash
python3 nbodies_grid_numba_2.py data/galaxy_1000 0.001 5 5 5
```

Paramètres : fichier_galaxie, dt, nx, ny, nz

### Version Numba Parallèle (Point 2)

```bash
NUMBA_NUM_THREADS=4 python3 nbodies_grid_numba_2.py data/galaxy_5000 0.001 10 10 10
```

Définissez NUMBA_NUM_THREADS avant d'exécuter. Valeurs typiques : 1, 2, 4, 8.

### Version MPI (Point 3 et 4)

**Benchmark (sans visualisation):**
```bash
mpirun -np 2 ./mi_examen_env/bin/python3 nbodies_grid_numba_2_para_v2.py data/galaxy_5000 0.001 10 10 1 1
```

**Visualisation interactive:**
```bash
mpirun -np 4 ./mi_examen_env/bin/python3 nbodies_grid_numba_2_para_v2.py data/galaxy_5000 0.001 10 10 0
```

Paramètres : np (nombre de processus), fichier_galaxie, dt, nx, ny, nz, benchmark_steps (0=visualisation, >0=benchmark)

---

## Fichiers Principaux

**nbodies_grid_numba_2_para_v2.py:** Implémentation MPI distribuée avec domain decomposition. Master coordonne, workers calculent l'accélération uniquement dans les cellules assignées.

**visualizer3d_sans_vbo.py:** Moteur graphique 3D sans VBO (compatible avec MPI). Importé conditionnellement dans Rank 0 pour éviter l'initialisation du contexte OpenGL dans les processus workers.

**nbodies_grid_numba_2.py:** Version séquentielle optimisée (référence pour les Points 1-2).

**Données:** galaxy_1000, galaxy_5000 (format : x y z vx vy vz mass par ligne)

---

## Pour aller plus loin : Algorithme Barnes-Hut

### Algorithme

L'algorithme de Barnes et Hut réduit la complexité de O(N²) à O(N log N) par un Octree hiérarchique. Au lieu de calculer la force avec chaque corps individuellement, il groupe les corps distants dans les nœuds de l'arbre et calcule leur interaction comme s'il s'agissait d'un seul corps au centre de masse du nœud.

Construction de l'Octree :
1. Partitionner récursivement l'espace en 8 octants
2. Assigner les corps aux feuilles
3. Calculer la masse et le COM de chaque nœud interne

Calcul des forces (par corps) :
- Si nœud est une feuille proche : itérer les corps internes
- Si nœud est loin (critère : size/distance < θ) : utiliser le COM du nœud
- Si nœud est proche : descendre récursivement

### Parallélisation en MPI

Pour distribuer le calcul de l'Octree en MPI :

1. Construction parallèle : Diviser les sous-arbres entre les processus. Chaque processus construit son sous-arbre local, puis synchroniser via Allreduce/Broadcast pour la structure complète.

2. Localité améliorée : Utiliser les Courbes de Peano-Hilbert pour mapper l'Octree aux processus. Les processus adjacents sur la courbe gèrent des régions adjacentes dans l'espace → meilleure localité du cache, moins de communication MPI.

3. Réduction des données : Synchroniser uniquement le COM et la masse des nœuds (O(N log N) au lieu de O(N) positions complètes).

Estimation : Speedup potentiel 50x+ pour N > 100k corps, mais nécessite une implémentation complexe de l'Octree parallèle.

---

## Notes Techniques

**Unités:** année-lumière, masse solaire, année. Constante G = 1.560339e-13.

**Optimisation R=2:** compute_acceleration itère uniquement ~27 cellules voisines (rayon R=2) au lieu de 400 cellules totales = 14.8x moins de travail.

**Domain decomposition:** Chaque rank assigne un segment contigu dans la dimension X, minimise la communication de frontière.

**Synchronisation:** Broadcast (initialiser), Gatherv (collecter les positions), Allreduce (synchroniser masques/COM des cellules), Barrier (attendre en fin d'itération).

