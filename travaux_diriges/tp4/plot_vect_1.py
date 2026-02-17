import matplotlib.pyplot as plt

# Données : Nombre de processus
processus = [1, 2, 3, 4, 5, 6]

# Temps de calcul (Moyenne Calcul) pour chaque résolution (Version Vectorisée)
t_200x400 = [0.001176, 0.002026, 0.002427, 0.002903, 0.003488, 0.004327]
t_400x800 = [0.001306, 0.002088, 0.002560, 0.003010, 0.003809, 0.004735]
t_800x1600 = [0.001482, 0.002078, 0.003213, 0.003062, 0.003908, 0.004686]

plt.figure(figsize=(12, 7))

# Tracé des courbes
plt.plot(processus, t_200x400, marker='o', label='200x400 (Vectorisé)', color='#3498db')
plt.plot(processus, t_400x800, marker='s', label='400x800 (Vectorisé)', color='#2ecc71')
plt.plot(processus, t_800x1600, marker='^', label='800x1600 (Vectorisé)', color='#e74c3c')

# Configuration du graphique
plt.title('Analyse de la Version Vectorisée et Parallélisée', fontsize=14, fontweight='bold')
plt.xlabel('Nombre de Processus', fontsize=12)
plt.ylabel('Temps moyen de Calcul (secondes)', fontsize=12)
plt.xticks(processus)
plt.grid(True, linestyle='--', alpha=0.6)

# Ajouter de l'espace pour la légende
plt.ylim(0, max(t_800x1600) * 1.5)
plt.legend(loc='upper left', title="Résolutions")

# Annotation importante pour le rapport
plt.text(1.5, 0.0035, "Note : Le temps augmente légèrement avec\nles processus à cause du surcoût (overhead)\nde synchronisation sur un calcul déjà très rapide.", 
         bbox=dict(facecolor='orange', alpha=0.2))

plt.tight_layout()
plt.savefig('performance_vectorisee_parallele.png')
plt.show()