# aggregators.py
import numpy as np

def flatten_weights(weights_list):
    """Aplatit une liste de paramètres en vecteurs 1D."""
    return [
        np.concatenate([w.flatten() for w in weights])
        for weights in weights_list
    ]

def krum(weights_list, f=1):
    """
    Sélectionne le client le plus représentatif.
    f = nombre de clients malveillants supposés
    """
    flat = flatten_weights(weights_list)
    n = len(flat)
    scores = []
    
    for i in range(n):
        distances = sorted([
            np.sum((flat[i] - flat[j]) ** 2)
            for j in range(n) if j != i
        ])
        # On garde les n-f-1 plus proches voisins
        scores.append(sum(distances[:n - f - 1]))
    
    best_idx = int(np.argmin(scores))
    return weights_list[best_idx], best_idx

def trimmed_mean(weights_list, trim=1):
    flat = flatten_weights(weights_list)
    n = len(flat)
    
    # Vérification : trim doit laisser au moins 2 valeurs
    assert n - 2 * trim >= 2, \
        f"Trop peu de clients ({n}) pour trim={trim}. Il faut au moins {2*trim+2} clients."
    
    stacked = np.stack(flat)
    stacked_sorted = np.sort(stacked, axis=0)
    trimmed = stacked_sorted[trim : n - trim]
    result_flat = np.mean(trimmed, axis=0)
    
    # Redonner la forme originale
    result = []
    idx = 0
    for w in weights_list[0]:
        size = w.size
        result.append(result_flat[idx:idx+size].reshape(w.shape))
        idx += size
    
    return result