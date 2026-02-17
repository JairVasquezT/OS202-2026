#!/bin/bash

# Définition des résolutions (Lignes Colonnes)
# Note : Pour flat, on respecte l'ordre attendu par ton script
RELS=("200 400" "400 800" "800 1600")

echo "--- Début des tests automatisés (Jeu de la Vie) ---"

# 1. Tests pour game_of_life.py (Version Séquentielle)
for res in "${RELS[@]}"; do
    echo "Exécution séquentielle (v0) - Résolution: $res"
    python3 game_of_life.py flat $res
    python3 game_of_life.py flat $res
done

# 2. Tests pour game_of_life_1.py (Mémoire Partagée / Version 1)
for res in "${RELS[@]}"; do
    echo "Exécution v1 - Résolution: $res"
    python3 game_of_life_1.py flat $res
    python3 game_of_life_1.py flat $res
done

# 3. Tests pour game_of_life_2.py (Décomposition de domaine)
# On boucle sur le nombre de cœurs (1 à 6)
for c in {3..6}; do
    for res in "${RELS[@]}"; do
        echo "Exécution v2 - Cores: $c - Résolution: $res"
        python3 game_of_life_2.py flat $c $res
        python3 game_of_life_2.py flat $c $res
    done
done

#4. Tests pour game_of_life_vect.py (Version Vectorisée)
for res in "${RELS[@]}"; do
    echo "Exécution séquentielle vectorisée - Résolution: $res"
    python3 game_of_life_vect.py flat $res
    python3 game_of_life_vect.py flat $res
done

#5. Tests pour game_of_life_vect_1.py (Version Vectorisée)
for c in {2..6}; do
    for res in "${RELS[@]}"; do
        echo "Exécution v2 - Cores: $c - Résolution: $res"
        python3 game_of_life_vect_1.py flat $res $c
        python3 game_of_life_vect_1.py flat $res $c
    done
done

echo "--- Tests terminés ! Vérifie tes fichiers .txt ---"
