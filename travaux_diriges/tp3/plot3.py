import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION DES DONNÉES ---
ranks = np.arange(6)
cores = [1, 2, 3, 4, 5, 6]

# Données de charge (Nombre d'éléments)
charge_uni = [167079, 166818, 166960, 166097, 166481, 166561]
charge_norm = [22707, 135889, 340905, 341664, 136195, 22636]
charge_expo = [283130, 203904, 145794, 104174, 74300, 188694]

# Données de temps de tri local (Sort Time)
temps_uni = [0.0527, 0.0460, 0.0528, 0.0458, 0.0479, 0.0451]
temps_norm = [0.0093, 0.0494, 0.1514, 0.1289, 0.0498, 0.0078]
temps_expo = [0.1077, 0.0643, 0.0470, 0.0424, 0.0235, 0.0520]

# Données de Speedup Global (Temps total de l'algorithme)
t_total_uni = [0.3329, 0.2160, 0.1792, 0.1345, 0.1437, 0.1241]
t_total_norm = [0.3313, 0.5485, 0.6705, 0.5216, 0.5827, 0.5946]
s_uni = [t_total_uni[0] / t for t in t_total_uni]
s_norm = [t_total_norm[0] / t for t in t_total_norm]

# --- GRAPHIQUE 1 : RÉPARTITION DE LA CHARGE ---
plt.figure(figsize=(10, 6))
width = 0.25
plt.bar(ranks - width, charge_uni, width, label='Uniforme', color='skyblue', edgecolor='black')
plt.bar(ranks, charge_norm, width, label='Normale', color='salmon', edgecolor='black')
plt.bar(ranks + width, charge_expo, width, label='Exponentielle', color='lightgreen', edgecolor='black')
plt.title('Répartition de la Charge (Nombre d\'éléments) par Processus', fontsize=12, fontweight='bold')
plt.xlabel('Rang MPI')
plt.ylabel('Nombre d\'éléments')
plt.xticks(ranks)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('repartition_charge.png', dpi=300)
plt.show() # Affiche le premier
print("Graphique 1 enregistré : repartition_charge.png")

# --- GRAPHIQUE 2 : TEMPS DE TRI LOCAL ---
plt.figure(figsize=(10, 6))
plt.bar(ranks - width, temps_uni, width, label='Uniforme', color='skyblue', edgecolor='black')
plt.bar(ranks, temps_norm, width, label='Normale', color='salmon', edgecolor='black')
plt.bar(ranks + width, temps_expo, width, label='Exponentielle', color='lightgreen', edgecolor='black')
plt.title('Temps de Tri Local par Processus (p=6)', fontsize=12, fontweight='bold')
plt.xlabel('Rang MPI')
plt.ylabel('Temps (secondes)')
plt.xticks(ranks)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('temps_tri_local.png', dpi=300)
plt.show() # Affiche le deuxième
print("Graphique 2 enregistré : temps_tri_local.png")

# --- GRAPHIQUE 3 : ANALYSE DU SPEEDUP ---
plt.figure(figsize=(10, 6))
plt.plot(cores, s_uni, 'o-', linewidth=2, label='Distribution Uniforme')
plt.plot(cores, s_norm, 's-', linewidth=2, label='Distribution Normale', color='red')
plt.plot(cores, cores, '--', color='gray', label='Speedup Idéal')
plt.title('Analyse du Speedup Global (Accélération)', fontsize=12, fontweight='bold')
plt.xlabel('Nombre de Processus (p)')
plt.ylabel('Speedup (T1/Tp)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)
plt.savefig('speedup_global.png', dpi=300)
plt.show() # Affiche le troisième
print("Graphique 3 enregistré : speedup_global.png")