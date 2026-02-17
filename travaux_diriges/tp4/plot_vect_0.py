import matplotlib.pyplot as plt
import numpy as np

# Données pour la version VECTORISÉE
resolutions = ['200x400', '400x800', '800x1600']
temps_calcul = [0.007521, 0.007153, 0.007133]
temps_affichage = [0.053051, 0.057656, 0.061252]

# Configuration de l'emplacement des barres
x = np.arange(len(resolutions))
largeur = 0.5

# Création de la figure
fig, ax = plt.subplots(figsize=(10, 7))

# Création des barres empilées
barre_calcul = ax.bar(x, temps_calcul, largeur, label='Temps de Calcul (Vectorisé NumPy)', color='#3498db')
barre_affichage = ax.bar(x, temps_affichage, largeur, bottom=temps_calcul, 
                          label='Temps d\'Affichage (Pygame)', color='#e67e22')

# Ajout des étiquettes et du texte
ax.set_xlabel('Résolution de la grille (Modèle "flat")', fontsize=12)
ax.set_ylabel('Temps moyen par itération (secondes)', fontsize=12)
ax.set_title('Performance : Version Parallèle Vectorisée', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(resolutions)
ax.legend()

# Ajout des valeurs totales au-dessus de chaque barre
for i in range(len(resolutions)):
    total = temps_calcul[i] + temps_affichage[i]
    ax.text(i, total + 0.002, f'{total:.3f}s', ha='center', fontweight='bold')

# Ajout d'une grille pour la lisibilité
ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()

# Sauvegarde de l'image
plt.savefig('performance_vectorisee.png')

# Affichage
plt.show()