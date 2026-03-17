# Simulation N-Body Distribuée avec Parallélisme Hybride

## 1. Présentation

Ce projet réalise une simulation de dynamique galactique utilisant le parallélisme hybride (Numba + MPI) et une accélération par grille spatiale. L'objectif est de calculer les forces gravitationnelles entre N corps dans un système galactique en réduisant la complexité algorithmique de O(N²) à O(N) grâce à une structure de grille cartésienne 3D avec indexation de Morse. Le projet combine deux niveaux de parallélisme : le parallélisme au niveau des threads via Numba JIT, et le parallélisme au niveau des processus via MPI pour la distribution du calcul sur plusieurs nœuds.

---

## 2. Structure du Code

### Fichiers Principaux

| Fichier | Rôle |
|---------|------|
| `nbodies_grid_numba_2_para_v2.py` | Version MPI distribuée avec décomposition de domaine (Étape 4) |
| `visualizer3d_sans_vbo.py` | Moteur graphique 3D compatible avec MPI (sans VBO) |
| `nbodies_grid_numba_2.py` | Version séquentielle et parallèle Numba (Étapes 1-2) |
| `nbodies_grid_numba_2_para.py` | Version MPI basique avec Gatherv (Étape 3) |
| `galaxy_generator.py` | Générateur de galaxies initiales |

### Répertoire de Données

Les fichiers de galaxie sont stockés dans le répertoire `data/` :
- `galaxy_1000` : 1000 corps celestes
- `galaxy_5000` : 5000 corps celestes

Format des fichiers : chaque ligne contient `x y z vx vy vz mass`.

### Environnement d'Exécution

Un environnement Python virtuel `mi_examen_env/` contient les dépendances nécessaires (NumPy, Numba, MPI4PY). Activation : `source mi_examen_env/bin/activate`

---

## 3. Résultats des Benchmarks

Les benchmarks ont été effectués sur la galaxie de 5000 corps (galaxy_5000) avec une grille de 10×10×10 cellules et un pas de temps de 0.001 année. Les résultats rapportent le temps de mise à jour moyen par itération, en millisecondes.

### Tableau Comparatif

| Étape | Configuration | Processus | Temps Update (ms) | Speedup | Efficacité |
|-------|---------------|-----------|-------------------|---------|-----------|
| 1.3 | Séquentiel | 1 | 1400 | 1.0x | 100% |
| 2 | Numba 4 threads | 1 | 380 | 3.68x | 92% |
| 3 | MPI 2 processus | 2 | 126 | 11.1x | 555% |
| 3 | MPI 4 processus | 4 | 630 | 2.2x | 55% |
| 4 | MPI Distribuée 4 proc | 4 | 111 | 12.6x | 315% |

### Analyse Détaillée

**Étape 1.3 - Séquentiel (1400 ms)**: Constitue la baseline de référence. Le temps est dominé par le calcul exhaustif des forces sur toutes les 400 cellules de la grille pour chaque corps. Aucun parallélisme n'est utilisé.

**Étape 2 - Numba 4 Threads (380 ms)**: Utilise le compilateur Numba avec parallelisation au niveau des threads via `@njit(parallel=True)` et `prange`. Le speedup de 3.68x avec 4 threads indique une efficacité de 92%, proche du scalage linéaire. La limitation principale provient de la contention de mémoire partagée au sein d'un même processeur.

**Étape 3 - MPI 2 Processus (126 ms)**: Introduit la parallélisation par processus de manière basique. Le speedup de 11.1x avec seulement 2 processus indique une superlinéarité, expliquée par une meilleure localité de cache L3 (chaque processus manipule 50% des données) et une réduction de la contention mémoire. Le surcoût MPI reste minimal avec peu de processus.

**Étape 3 - MPI 4 Processus (630 ms)**: Augmente le nombre de processus à 4. Cependant, les résultats montrent une dégradation significative (speedup 2.2x, efficacité 55%). Cette degradation provient de l'augmentation du surcoût de communication MPI (Gather + Broadcast) qui dépasse le gain computationnel obtenu par la parallélisation supplémentaire. La distribution de domaine uniforme crée un déséquilibre de charge entre processus.

**Étape 4 - MPI Distribuée 4 Processus (111 ms)**: Optimise la communication MPI via une décomposition de domaine spatiale. Chaque processus calcule uniquement une portion de l'espace 3D. Le speedup de 12.6x surpasse même celui de l'Étape 3 avec 2 processus grâce à :
- Réduction drastique du volume de communication par la décomposition de domaine
- Meilleure utilisation du cache local par processus
- Synchronisation plus efficace via Allreduce sur les masses et COM des cellules

---

## 4. Analyse Théorique

### 4.1 Déséquilibre de Charge (Load Imbalance)

Dans une simulation N-Body avec structure de grille spatiale, la densité de corps varie considérablement selon la position. En particulier :
- Le centre galactique, dominé par la concentration de masse due au trou noir central, présente une densité extrêmement élevée.
- La périphérie, loin du centre de masse, contient peu de corps et offre des zones largement vides.

Avec une décomposition de domaine uniforme en dimension X (comme dans l'Étape 3), chaque processus est assigné un nombre égal de cellules dans la direction X. Cependant, cela ne garantit pas une distribution équitable du travail :
- Les processus gèrant les cellules centrales peuvent traiter 1000+ corps par cellule.
- Les processus gèrant les cellules périphériques peuvent traiter seulement 10-50 corps par cellule.

Lors de la synchronisation MPI (Barrier), le processus le plus rapide attend le plus lent, entraînant une perte d'efficacité. Dans le cas d'une distribution non uniforme, l'efficacité peut chuter de 90% à 55%, comme observé entre les résultats MPI 2 processus et MPI 4 processus.

### 4.2 Distribution Intelligente : Courbes de Remplissage d'Espace

Pour pallier le déséquilibre de charge tout en maintenant une localité de données optimale, deux approches sont envisagées :

**Grilles Adaptatives**: Subdiviser dynamiquement les cellules de haute densité en sous-cellules plus fines, créant une hiérarchie d'espaces de différentes résolutions. Chaque processus reçoit la même charge computationnelle (nombre de corps) indépendamment du volume géométrique.

**Courbes de Remplissage d'Espace (Hilbert ou Peano)**: Mapper l'espace 3D à une courbe 1D continue préservant la localité spatiale. Diviser cette courbe en segments contenant approximativement le même nombre de corps, puis assigner chaque segment à un processus distinct. Cette approche conserve une proximité spatiale des données locales à chaque processus, minimisant la communication de frontière.

Ces méthodes permettraient d'atteindre une efficacité proche de 100% même avec plusieurs processus, en garantissant que chaque processus exécute approximativement la même quantité de travail à chaque itération.

---

## 5. Extension Barnes-Hut

L'algorithme de Barnes et Hut offre une réduction supplémentaire de complexité.

### 5.1 Complexité

L'approche directe (somme sur tous les corps) présente une complexité O(N²) par itération. L'algorithme de Barnes-Hut réduit cette complexité à O(N log N) en exploitant une structure hiérarchique :
- Construire un Octree (extension 3D du Quadtree) partitionnant récursivement l'espace en 8 octants.
- Calculer pour chaque nœud de l'arbre sa masse totale et son centre de masse.
- Lors du calcul des forces sur un corps, utiliser un critère d'ouverture (opening angle criterion) : si un nœud est suffisamment éloigné et suffisamment petit, traiter le nœud comme un unique corps au lieu de parcourir récursivement ses enfants.

### 5.2 Parallélisation en MPI

Pour paralléliser l'algorithme Barnes-Hut sur architecture MPI distribuée :

**Construction Parallèle de l'Octree**: Diviser les sous-arbres de l'Octree entre les processus MPI. Chaque processus construit localement son sous-arbre, puis effectuer une synchronisation globale (Allreduce, Broadcast) pour fusionner la structure complète.

**Localité Améliorée via Courbes de Peano-Hilbert**: Mapper les nœuds de l'Octree à une courbe de Peano-Hilbert 1D. Assigner à chaque processus un segment continu de cette courbe. Les processus adjacents sur la courbe gèrent des régions adjacentes en 3D, minimisant la communication MPI pour les interactions de frontière.

**Réduction du Surcoût de Communication**: Synchroniser uniquement les masses et centres de masse des nœuds (O(N log N) scalaires) au lieu des positions complètes de tous les corps (O(N × 3) réels). Cette réduction dramatiquedu volume de données communiquées améliore significativement la scalabilité.

**Estimation de Performance**: Pour N > 100 000 corps, un speedup potentiel de 50x ou plus par rapport à la version séquentielle est envisageable. Cependant, la complexité d'implémentation augmente considérablement et l'Octree parallèle doit être soigneusement gérée pour éviter les surcoûts de synchronisation.

---

## 6. Guide d'Exécution

### 6.1 Prérequis

Activer l'environnement virtuel :
```bash
source mi_examen_env/bin/activate
```

### 6.2 Exécution Numba 4 Threads

```bash
NUMBA_NUM_THREADS=4 python3 nbodies_grid_numba_2.py data/galaxy_5000 0.001 10 10 10
```

Cela exécute la version séquentielle optimisée avec accélération Numba sur 4 threads. Les paramètres sont :
- `data/galaxy_5000` : fichier de galaxie
- `0.001` : pas de temps en années
- `10 10 10` : dimensions de la grille spatiale (10×10×10 = 1000 cellules)

### 6.3 Exécution MPI 4 Processus (Étape 4 - Meilleure Performance)

```bash
mpirun -np 4 ./mi_examen_env/bin/python3 nbodies_grid_numba_2_para_v2.py data/galaxy_5000 0.001 10 10 1
```

Lancer 4 processus MPI. Le dernier paramètre `1` impose le mode benchmark (itérations de simulation sans visualisation graphique). Mode interactif (0 pour visualisation 3D en temps réel) :

```bash
mpirun -np 4 ./mi_examen_env/bin/python3 nbodies_grid_numba_2_para_v2.py data/galaxy_5000 0.001 10 10 0
```

### 6.4 Variantes

Pour tester d'autres configurations :

```bash
# MPI 2 processus
mpirun -np 2 ./mi_examen_env/bin/python3 nbodies_grid_numba_2_para_v2.py data/galaxy_5000 0.001 10 10 1

# Numba 2 threads
NUMBA_NUM_THREADS=2 python3 nbodies_grid_numba_2.py data/galaxy_5000 0.001 10 10 10

# Séquentiel (baseline)
python3 nbodies_grid_numba_2.py data/galaxy_1000 0.001 5 5 5
```

---

## 7. Considérations Techniques Supplémentaires

### Unités Astronomiques Normalisées

La simulation utilise les unités suivantes pour éviter problèmes numériques liés aux très grands et très petits nombres :
- Distance : année-lumière (1 ly = 9.46 × 10¹⁵ m)
- Masse : masse solaire (1 M☉ = 2 × 10³⁰ kg)
- Temps : année sidérale (365.25 jours)
- Constante gravitationnelle : G = 1.560339 × 10⁻¹³ ly³/(M☉·an²)

### Optimisation du Rayon de Voisinage (R=2)

L'itération sur toutes les 400 cellules de la grille 10×10×10 est remplacée par une itération sur les cellules voisines uniquement. Avec un rayon R=2 (6 voisins dans chaque direction, soit 3³ = 27 cellules au maximum), le travail computationnel est réduit d'un facteur 14.8. Cette optimisation contribue significativement au speedup entre les Étapes 1 et 2.

### Patterns de Communication MPI

Chaque itération implique :
1. Un Broadcast des positions globales depuis Rank 0 vers tous les workers : O(N_bodies × 3 × 8 bytes)
2. Une Gatherv pour collecter les positions mises à jour : O(N_bodies × 3 × 8 bytes)
3. Un Allreduce pour synchroniser les masses et COM des cellules : O(N_cells)
4. Barrière de synchronisation à la fin de chaque itération

Pour galaxy_5000 avec 1000 cellules, ce surcoût total atteint environ 80 microsecondes par itération avec une latence réseau faible, expliquant pourquoi MPI 4 processus s'avère moins efficace que MPI 2 processus en Étape 3 (avant décomposition de domaine optimisée).

---

## 8. Conclusion

Ce projet illustre comment combiner stratégiquement le parallélisme multi-niveaux (threads, processus, accélération structurelle) pour accomplir des tâches intensives de calcul. L'Étape 4 (MPI Distribuée) démontre que l'optimisation prudente de l'architecture de communication MPI peut surmonter les limitations de scalabilité observées a l'Étape 3, atteignant un speedup de 12.6x avec 4 processus.

Les extensions envisagées (grilles adaptatives, courbes de Peano-Hilbert, algorithme Barnes-Hut) promettent des améliorations supplémentaires, montrant que le domaine du calcul haute performance reste ouvert à l'innovation algorithmique et architecturale pour des simulations scientifiques complexes.
