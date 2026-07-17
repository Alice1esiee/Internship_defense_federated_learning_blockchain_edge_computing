"""
compression.py
Compression des gradients/poids par quantization pour l'Expérience 5.
Compatible avec le pipeline Flower de la collègue (NumPy arrays).

Deux méthodes disponibles :
  - quantize_weights()   : quantization uniforme 8-bit ou 4-bit
  - dequantize_weights() : reconstruction des poids compressés
  - compress_weights()   : interface unique (quantize + métadonnées)
  - decompress_weights() : interface unique (dequantize)

Métriques :
  - compression_ratio()  : ratio de compression atteint
  - quantization_error() : erreur L2 entre poids originaux et reconstruits
"""

import numpy as np
from dataclasses import dataclass, field


# ── Structures de données ─────────────────────────────────────────────────────

@dataclass
class CompressedWeights:
    """Conteneur pour les poids compressés d'un client."""
    quantized: list[np.ndarray]   # poids quantizés (int16, valeurs 0..255 ou 0..15)
    mins: list[float]             # min par couche (pour dequantize)
    maxs: list[float]             # max par couche (pour dequantize)
    bits: int                     # 8 ou 4
    original_shapes: list[tuple]  # shapes originales
    original_dtype: str           # dtype original (float32)


# ── Quantization uniforme ─────────────────────────────────────────────────────

def quantize_weights(weights: list[np.ndarray],
                     bits: int = 8) -> CompressedWeights:
    """
    Quantization uniforme par couche (min-max scaling).

    bits=8 → int8  (256 niveaux)  — bon compromis précision/compression
    bits=4 → int4 simulé en int8  (16 niveaux) — compression max, perte plus forte

    Retourne un objet CompressedWeights.
    """
    if bits not in (4, 8):
        raise ValueError(f"bits doit être 4 ou 8, reçu : {bits}")

    num_levels = 2 ** bits - 1  # 255 pour 8-bit, 15 pour 4-bit
    quantized = []
    mins = []
    maxs = []
    shapes = []

    for layer in weights:
        shapes.append(layer.shape)
        w_min = float(layer.min())
        w_max = float(layer.max())
        mins.append(w_min)
        maxs.append(w_max)

        # Évite division par zéro si tous les poids sont identiques
        scale = w_max - w_min
        if scale == 0.0:
            scale = 1.0

        # Normalise dans [0, num_levels], arrondi en entier
        # On stocke en int16 pour éviter l'overflow int8 (0-255 > 127)
        q = np.round((layer - w_min) / scale * num_levels).astype(np.int16)
        quantized.append(q)

    return CompressedWeights(
        quantized=quantized,
        mins=mins,
        maxs=maxs,
        bits=bits,
        original_shapes=shapes,
        original_dtype="float32",
    )


def dequantize_weights(compressed: CompressedWeights) -> list[np.ndarray]:
    """
    Reconstruit les poids float32 à partir d'un objet CompressedWeights.
    """
    num_levels = 2 ** compressed.bits - 1
    weights = []

    for q, w_min, w_max, shape in zip(
        compressed.quantized, compressed.mins,
        compressed.maxs, compressed.original_shapes
    ):
        scale = w_max - w_min
        if scale == 0.0:
            scale = 1.0
        # Reconstruction : dénormalise vers l'espace float32 original
        w = (q.astype(np.float32) / num_levels) * scale + w_min  # int16 → float32
        weights.append(w.reshape(shape))

    return weights


# ── Interface haut niveau ─────────────────────────────────────────────────────

def compress_weights(weights: list[np.ndarray],
                     bits: int = 8) -> CompressedWeights:
    """
    Compresse les poids d'un client avant envoi au serveur.
    À appeler côté client juste avant fl.common.ndarrays_to_parameters().

    Usage :
        compressed = compress_weights(get_parameters(model), bits=8)
        # sérialisation : on envoie compressed.quantized au serveur
    """
    return quantize_weights(weights, bits=bits)


def decompress_weights(compressed: CompressedWeights) -> list[np.ndarray]:
    """
    Décompresse les poids reçus par le serveur avant agrégation.
    À appeler côté serveur dans aggregate_fit().

    Usage :
        weights = decompress_weights(compressed)
        # puis passe weights à krum() / trimmed_mean() / fedavg()
    """
    return dequantize_weights(compressed)


# ── Métriques ─────────────────────────────────────────────────────────────────

def compression_ratio(weights: list[np.ndarray],
                      compressed: CompressedWeights) -> float:
    """
    Ratio de compression = taille originale / taille compressée.
    > 1 signifie compression effective.

    float32 = 4 octets/élément
    int8    = 1 octet/élément
    int4 simulé en int8 = 1 octet/élément (idéalement 0.5, mais stocké en int8)
    """
    original_bytes = sum(w.nbytes for w in weights)  # float32 → 4 bytes
    compressed_bytes = sum(q.nbytes for q in compressed.quantized)  # int8 → 1 byte
    # +métadonnées légères (mins, maxs) : négligeables
    return original_bytes / compressed_bytes


def quantization_error(original: list[np.ndarray],
                       reconstructed: list[np.ndarray]) -> dict:
    """
    Erreur de quantization entre poids originaux et reconstruits.

    Métriques retournées :
    - rmse_normalized : RMSE / std(original) — indépendant de la taille des couches
    - mean_max_error  : erreur max moyenne par couche
    - per_layer       : détail par couche
    """
    errors = []
    for orig, recon in zip(original, reconstructed):
        orig_f = orig.astype(np.float32).ravel()
        recon_f = recon.astype(np.float32).ravel()
        diff = orig_f - recon_f
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        std = float(np.std(orig_f)) + 1e-8
        rmse_norm = rmse / std          # sans unité, comparable entre couches
        max_err = float(np.abs(diff).max())
        errors.append({"rmse": rmse, "rmse_normalized": rmse_norm, "max": max_err})

    mean_rmse_norm = float(np.mean([e["rmse_normalized"] for e in errors]))
    mean_max = float(np.mean([e["max"] for e in errors]))

    return {
        "per_layer": errors,
        "mean_l2_normalized": mean_rmse_norm,   # gardé pour compatibilité
        "mean_rmse_normalized": mean_rmse_norm,
        "mean_max_error": mean_max,
    }


def weights_size_bytes(weights: list[np.ndarray]) -> int:
    """Taille totale en octets des poids (float32)."""
    return sum(w.nbytes for w in weights)


# ── Test rapide ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Test compression.py ===\n")

    # Simulation de poids CNN (comme SimpleCNN de l'Exp 4)
    fake_weights = [
        np.random.randn(32, 3, 3, 3).astype(np.float32),   # Conv1 weights
        np.random.randn(32).astype(np.float32),             # Conv1 bias
        np.random.randn(64, 32, 3, 3).astype(np.float32),  # Conv2 weights
        np.random.randn(64).astype(np.float32),             # Conv2 bias
        np.random.randn(256, 4096).astype(np.float32),      # FC1 weights
        np.random.randn(256).astype(np.float32),            # FC1 bias
        np.random.randn(10, 256).astype(np.float32),        # FC2 weights
        np.random.randn(10).astype(np.float32),             # FC2 bias
    ]

    original_size = weights_size_bytes(fake_weights)
    print(f"Taille originale (float32) : {original_size / 1024:.1f} KB")

    for bits in [8, 4]:
        print(f"\n── Quantization {bits}-bit ──")
        compressed = compress_weights(fake_weights, bits=bits)
        reconstructed = decompress_weights(compressed)

        ratio = compression_ratio(fake_weights, compressed)
        error = quantization_error(fake_weights, reconstructed)
        comp_size = sum(q.nbytes for q in compressed.quantized)

        print(f"  Taille compressée    : {comp_size / 1024:.1f} KB")
        print(f"  Ratio de compression : {ratio:.1f}x")
        print(f"  Erreur L2 normalisée : {error['mean_l2_normalized']:.6f}")
        print(f"  Erreur max moyenne   : {error['mean_max_error']:.6f}")
