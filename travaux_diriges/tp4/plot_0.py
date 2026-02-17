import matplotlib.pyplot as plt

# Données extraites de tes fichiers de résultats (.txt)
# Nous comparons les différentes résolutions pour le pattern "flat"
resolutions = ['200x400', '400x800', '800x1600']
temps_calcul = [0.899401, 0.895839, 0.938199]
temps_affichage = [0.152213, 0.165358, 0.163051]

# Position des barres sur l'axe X
x = range(len(resolutions))

# Création du graphique
plt.figure(figsize=(10, 6))

# Création des barres empilées (Stacked Bar Chart)
# Le temps de calcul en bas, le temps d'affichage par-dessus
plt.bar(x, temps_calcul, label='Temps de Calcul (Moyenne)', color='#3498db')
plt.bar(x, temps_affichage, bottom=temps_calcul, label='Temps d\'Affichage (Moyenne)', color='#e74c3c')

plt.ylim(0, 1.3)

# Ubicar la leyenda fuera del gráfico a la derecha
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Légende")

# Ajout des détails pour la clarté du rapport
plt.xlabel('Résolution de la grille (Nombre de cellules)')
plt.ylabel('Temps par itération (secondes)')
plt.title('Analyse des Performances : Temps de Calcul vs Affichage')
plt.xticks(x, resolutions)
plt.legend(loc='upper left')

# Ajout d'une grille horizontale pour faciliter la lecture des valeurs
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotation des valeurs totales au-dessus des barres
for i in range(len(resolutions)):
    total = temps_calcul[i] + temps_affichage[i]
    plt.text(i, total + 0.02, f'{total:.3f}s', ha='center', weight='bold')

# Sauvegarde du graphique pour l'insérer dans le rapport (Word, LaTeX, etc.)
plt.tight_layout()
plt.savefig('analyse_performance_flat.png',bbox_inches='tight')

# Affichage du graphique à l'écran
plt.show()