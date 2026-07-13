# test_aggregators.py
import numpy as np
from aggregators import krum, trimmed_mean

# Simuler 4 clients (plus réaliste pour trimmed mean)
client_0 = [np.array([1.0, 1.0, 1.0])]   # honnête
client_1 = [np.array([1.1, 0.9, 1.0])]   # honnête
client_2 = [np.array([1.0, 1.1, 1.0])]   # honnête
client_3 = [np.array([99., 99., 99.])]    # malveillant

weights_list = [client_0, client_1, client_2, client_3]

# Test Krum (f=1 : on suppose 1 malveillant)
result, chosen = krum(weights_list, f=1)
print(f"Krum a choisi le client {chosen}")
# Attendu : 0, 1 ou 2 (jamais 3)

# Test Trimmed Mean (trim=1 sur 4 clients)
result_tm = trimmed_mean(weights_list, trim=1)
print(f"Trimmed Mean résultat : {result_tm}")
# Attendu : proche de [1.03, 1.0, 1.0] (moyenne des 2 du milieu)

# Vérification visuelle
print("\n--- Vérification ---")
print(f"Client malveillant (3) exclu : {chosen != 3 if 'chosen' in dir() else 'voir Krum'}")