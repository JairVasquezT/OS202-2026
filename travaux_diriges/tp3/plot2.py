import matplotlib.pyplot as plt
import numpy as np

# Nombre de cœurs (processus MPI) utilisés pour les tests
cores = [1, 2, 3, 4, 5, 6]

# Temps total d'exécution (en secondes) mesuré pour chaque cas
# Données réelles pour 1 000 000 d'éléments
t_uniforme = [0.3329, 0.2160, 0.1792, 0.1345, 0.1437, 0.1241]
t_normal = [0.3313, 0.5485, 0.6705, 0.5216, 0.5827, 0.5946]

# Calcul du Speedup (S = T1 / Tp)
# T1 : Temps séquentiel (1 cœur) | Tp : Temps parallèle (p cœurs)
s_uniforme = [t_uniforme[0] / t for t in t_uniforme]
s_normal = [t_normal[0] / t for t in t_normal]

# Création de la figure
plt.figure(figsize=(9, 6))

# Tracer les courbes de performance
plt.plot(cores, s_uniforme, 'o-', linewidth=2, label='Distribution Uniforme (Optimale)')
plt.plot(cores, s_normal, 's-', linewidth=2, label='Distribution Normale (Déséquilibrée)', color='red')

# Tracer la ligne de Speedup Idéal (Loi d'Amdahl théorique)
plt.plot(cores, cores, '--', color='gray', alpha=0.7, label='Speedup Théorique (Idéal)')

# Configuration des axes et titres
plt.title('Comparaison du Speedup : Distribution Uniforme vs Normale', fontsize=14, fontweight='bold')
plt.xlabel('Nombre de cœurs (Processus MPI)', fontsize=12)
plt.ylabel('Speedup (T1 / Tp)', fontsize=12)

# Ajout d'une grille et d'une légende
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)

# Sauvegarde pour le rapport LaTeX
plt.savefig('speedup_comparatif_mpi.png', dpi=300)

# Affichage du graphique final
plt.show()