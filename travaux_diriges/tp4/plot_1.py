import matplotlib.pyplot as plt
import numpy as np

# Données extraites de tes résultats pour la version Parallélisée
resolutions = ['200x400', '400x800', '800x1600']
temps_calcul = [0.963021, 0.924002, 0.947864]
temps_affichage = [0.161785, 0.155927, 0.172118]

# Configuration de l'emplacement des barres
x = np.arange(len(resolutions))
largeur = 0.6

# Création de la figure
fig, ax = plt.subplots(figsize=(10, 7))

# Création des barres empilées
barre_calcul = ax.bar(x, temps_calcul, largeur, label='Temps de Calcul (Parallélisé)', color='#2ecc71')
barre_affichage = ax.bar(x, temps_affichage, largeur, bottom=temps_calcul, 
                          label='Temps d\'Affichage (Pygame)', color='#e67e22')

plt.ylim(0, 1.3)

# Ajout des étiquettes et du texte
ax.set_xlabel('Résolution de la grille (Modèle "flat")', fontsize=12)
ax.set_ylabel('Temps moyen par itération (secondes)', fontsize=12)
ax.set_title('Analyse des Performances : Modèle Parallélisé par Domaine (Sans Vecteurs)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(resolutions)
ax.legend()

# Ajout des valeurs totales au-dessus de chaque barre pour la précision
for i in range(len(resolutions)):
    total = temps_calcul[i] + temps_affichage[i]
    ax.text(i, total + 0.02, f'{total:.3f}s', ha='center', fontweight='bold')

# Ajout d'une grille pour la lisibilité
ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()

# Sauvegarde de l'image
plt.savefig('performance_parallelise_domaine.png')

# Affichage
plt.show()