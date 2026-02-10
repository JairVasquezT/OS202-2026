import matplotlib.pyplot as plt
import numpy as np

# Configuration des rangs MPI (0 à 5 pour 6 processus)
ranks = np.arange(6)

# Données réelles extraites de vos exécutions (np = 6, Taille = 1 000 000)
# Ces valeurs montrent le nombre d'éléments reçus par chaque cœur
uniforme = [167079, 166818, 166960, 166097, 166481, 166561]
normale = [22707, 135889, 340905, 341664, 136195, 22636]
exponentielle = [283130, 203904, 145794, 104174, 74300, 188694]

# Création de la figure avec trois sous-graphiques (côte à côte)
# sharey=True permet d'avoir la même échelle verticale pour comparer visuellement
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# --- Graphique 1 : Distribution UNIFORME ---
axs[0].bar(ranks, uniforme, color='skyblue', edgecolor='black')
axs[0].set_title('Distribution UNIFORME\n(Équilibrage Idéal)', fontsize=12, fontweight='bold')
axs[0].set_xlabel('Rang MPI')
axs[0].set_ylabel('Éléments traités (Charge)')
axs[0].grid(axis='y', linestyle='--', alpha=0.7)

# --- Graphique 2 : Distribution NORMALE ---
axs[1].bar(ranks, normale, color='salmon', edgecolor='black')
axs[1].set_title('Distribution NORMALE\n(Déséquilibre Central)', fontsize=12, fontweight='bold')
axs[1].set_xlabel('Rang MPI')
axs[1].grid(axis='y', linestyle='--', alpha=0.7)

# --- Graphique 3 : Distribution EXPONENTIELLE ---
axs[2].bar(ranks, exponentielle, color='lightgreen', edgecolor='black')
axs[2].set_title('Distribution EXPONENTIELLE\n(Biais à Gauche)', fontsize=12, fontweight='bold')
axs[2].set_xlabel('Rang MPI')
axs[2].grid(axis='y', linestyle='--', alpha=0.7)

# Ajustement de la disposition pour éviter les chevauchements
plt.tight_layout()

# Sauvegarde de l'image pour l'inclure dans le rapport LaTeX
plt.savefig('comparaison_distributions_mpi.png', dpi=300)

# Affichage du graphique
plt.show()



