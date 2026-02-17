import matplotlib.pyplot as plt

# Données organisées par nombre de processus (1 à 6)
processus = [1, 2, 3, 4, 5, 6]

# Temps de calcul pour chaque résolution
t_200x400 = [0.761727, 0.435999, 0.320240, 0.308454, 0.319123, 0.252274]
t_400x800 = [0.618420, 0.472915, 0.353632, 0.320537, 0.273168, 0.250959]
t_800x1600 = [0.761920, 0.466762, 0.368897, 0.320361, 0.323728, 0.262325]

# Création de la figure
plt.figure(figsize=(12, 7))

# Tracé des courbes pour chaque résolution
plt.plot(processus, t_200x400, marker='o', linestyle='-', linewidth=2, label='Résolution 200x400')
plt.plot(processus, t_400x800, marker='s', linestyle='-', linewidth=2, label='Résolution 400x800')
plt.plot(processus, t_800x1600, marker='^', linestyle='-', linewidth=2, label='Résolution 800x1600')

# Personnalisation des axes et titres en français
plt.title('Accélération du Calcul : Impact du nombre de Processus', fontsize=14, fontweight='bold')
plt.xlabel('Nombre de Processus (Cœurs)', fontsize=12)
plt.ylabel('Temps moyen de Calcul (secondes)', fontsize=12)
plt.xticks(processus)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(title="Grilles Testées (Modèle flat)")

plt.tight_layout()

# Sauvegarde de l'image pour le rapport
plt.savefig('analyse_scalabilite_calcul.png')

# Affichage du graphique
plt.show()