# test_realistic.py
import numpy as np
from aggregators import krum, trimmed_mean

# Simuler de vrais poids d'un petit réseau CNN (comme MNIST)
# Conv layer : (32, 1, 3, 3), Dense layer : (128, 32), Bias : (128,)
def make_honest_client(noise=0.05):
    return [
        np.random.normal(0.0, 0.1, (32, 1, 3, 3)).astype(np.float32),
        np.random.normal(0.0, 0.1, (128, 32)).astype(np.float32),
        np.random.normal(0.0, 0.1, (128,)).astype(np.float32),
    ]

def make_malicious_client(scale=10.0):
    return [
        np.random.normal(0.0, scale, (32, 1, 3, 3)).astype(np.float32),
        np.random.normal(0.0, scale, (128, 32)).astype(np.float32),
        np.random.normal(0.0, scale, (128,)).astype(np.float32),
    ]

np.random.seed(42)

# 3 clients honnêtes + 1 malveillant
clients = [
    make_honest_client(),   # client 0
    make_honest_client(),   # client 1
    make_honest_client(),   # client 2
    make_malicious_client() # client 3 - malveillant
]

print("=== Test avec vrais poids CNN ===\n")

# Test Krum
result_krum, chosen = krum(clients, f=1)
print(f"Krum a choisi le client {chosen}")
print(f"Client malveillant (3) exclu : {chosen != 3}")

# Test Trimmed Mean
result_tm = trimmed_mean(clients, trim=1)
print(f"\nTrimmed Mean :")
print(f"  Shape couche 1 : {result_tm[0].shape}")
print(f"  Shape couche 2 : {result_tm[1].shape}")
print(f"  Valeur max (doit être proche de 0) : {result_tm[0].max():.4f}")
print(f"  Valeur max client malveillant : {clients[3][0].max():.4f}")
print(f"  → Trimmed Mean a bien atténué le malveillant : "
      f"{abs(result_tm[0].max()) < abs(clients[3][0].max())}")