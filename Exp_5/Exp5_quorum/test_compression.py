"""
test_compression.py
Tests unitaires pour compression.py.
Valide la quantization, la dequantization et les métriques.

Lancer avec : python test_compression.py
"""

import numpy as np
import sys

# Import local
from compression import (
    compress_weights, decompress_weights,
    compression_ratio, quantization_error,
    weights_size_bytes, CompressedWeights,
)

PASS = "✓"
FAIL = "✗"
errors = []


def check(condition: bool, name: str, detail: str = ""):
    if condition:
        print(f"  {PASS} {name}")
    else:
        print(f"  {FAIL} {name}" + (f" — {detail}" if detail else ""))
        errors.append(name)


# ── Données de test ───────────────────────────────────────────────────────────
def make_weights(seed=0):
    rng = np.random.default_rng(seed)
    return [
        rng.standard_normal((32, 3, 3, 3)).astype(np.float32),
        rng.standard_normal((32,)).astype(np.float32),
        rng.standard_normal((64, 32, 3, 3)).astype(np.float32),
        rng.standard_normal((64,)).astype(np.float32),
        rng.standard_normal((256, 4096)).astype(np.float32),
        rng.standard_normal((256,)).astype(np.float32),
        rng.standard_normal((10, 256)).astype(np.float32),
        rng.standard_normal((10,)).astype(np.float32),
    ]


# ── Test 1 : sortie de compress_weights ──────────────────────────────────────
print("\n── Test 1 : structure de CompressedWeights ──")
weights = make_weights()
for bits in [8, 4]:
    compressed = compress_weights(weights, bits=bits)
    check(isinstance(compressed, CompressedWeights),
          f"bits={bits} : retourne CompressedWeights")
    check(len(compressed.quantized) == len(weights),
          f"bits={bits} : même nombre de couches ({len(weights)})")
    check(compressed.bits == bits,
          f"bits={bits} : attribut bits correct")
    check(all(q.dtype == np.int16 for q in compressed.quantized),
          f"bits={bits} : dtype int16")
    check(len(compressed.mins) == len(weights) and len(compressed.maxs) == len(weights),
          f"bits={bits} : mins/maxs présents")


# ── Test 2 : shapes conservées après dequantize ──────────────────────────────
print("\n── Test 2 : shapes conservées après décompression ──")
for bits in [8, 4]:
    compressed = compress_weights(weights, bits=bits)
    reconstructed = decompress_weights(compressed)
    shapes_ok = all(
        r.shape == o.shape
        for r, o in zip(reconstructed, weights)
    )
    check(shapes_ok, f"bits={bits} : shapes identiques après décompression")
    dtypes_ok = all(r.dtype == np.float32 for r in reconstructed)
    check(dtypes_ok, f"bits={bits} : dtype float32 après décompression")


# ── Test 3 : ratio de compression ────────────────────────────────────────────
print("\n── Test 3 : ratio de compression ──")
for bits in [8, 4]:
    compressed = compress_weights(weights, bits=bits)
    ratio = compression_ratio(weights, compressed)
    # float32 (4 bytes) → int16 (2 bytes) = ratio théorique 2x
    check(ratio >= 1.9,
          f"bits={bits} : ratio >= 1.9x (obtenu {ratio:.2f}x)")

original_size = weights_size_bytes(weights)
check(original_size > 0, f"weights_size_bytes > 0 ({original_size} bytes)")


# ── Test 4 : erreur de quantization ──────────────────────────────────────────
print("\n── Test 4 : erreur de quantization ──")
for bits in [8, 4]:
    compressed = compress_weights(weights, bits=bits)
    reconstructed = decompress_weights(compressed)
    err = quantization_error(weights, reconstructed)

    check("per_layer" in err and "mean_rmse_normalized" in err,
          f"bits={bits} : structure du dict d'erreur correcte")
    check(err["mean_rmse_normalized"] >= 0,
          f"bits={bits} : erreur RMSE normalisée >= 0")
    # Stocke pour comparaison après la boucle
    if bits == 8:
        err8 = err["mean_rmse_normalized"]
    else:
        err4 = err["mean_rmse_normalized"]

check(err4 > err8,
      f"4-bit moins précis que 8-bit (err4={err4:.5f} > err8={err8:.5f})")
# 8-bit : RMSE normalisée doit être faible (< 10% de la std)
check(err8 < 0.10,
      f"8-bit : RMSE normalisée < 10% (obtenu {err8*100:.3f}%)")


# ── Test 5 : cas limites ──────────────────────────────────────────────────────
print("\n── Test 5 : cas limites ──")

# Poids tous à zéro
zero_weights = [np.zeros((10, 10), dtype=np.float32)]
try:
    c = compress_weights(zero_weights, bits=8)
    r = decompress_weights(c)
    check(np.allclose(r[0], 0.0, atol=1e-5),
          "Poids tous nuls : reconstruction correcte")
except Exception as e:
    check(False, f"Poids tous nuls : exception inattendue — {e}")

# Poids constants (non nuls)
const_weights = [np.full((5, 5), 3.14, dtype=np.float32)]
try:
    c = compress_weights(const_weights, bits=8)
    r = decompress_weights(c)
    check(np.allclose(r[0], 3.14, atol=1e-3),
          "Poids constants : reconstruction correcte")
except Exception as e:
    check(False, f"Poids constants : exception inattendue — {e}")

# bits invalide
try:
    compress_weights(weights, bits=16)
    check(False, "bits=16 : doit lever ValueError")
except ValueError:
    check(True, "bits=16 : ValueError levée correctement")


# ── Test 6 : aller-retour compress → decompress ──────────────────────────────
print("\n── Test 6 : aller-retour (pipeline complet) ──")
small_weights = [np.random.randn(4, 4).astype(np.float32) for _ in range(3)]
for bits in [8, 4]:
    c = compress_weights(small_weights, bits=bits)
    r = decompress_weights(c)
    # L'erreur max doit être bornée par le pas de quantization
    step = 2.0 / (2 ** bits - 1)  # ~plage typique [-1, 1]
    err = quantization_error(small_weights, r)
    # erreur max bornée par le pas de quantization (plage ~[-3,3] pour randn → pas ~6/255 ≈ 0.024 en 8-bit)
    threshold = 0.5 if bits == 8 else 1.0
    check(err["mean_max_error"] < threshold,
          f"bits={bits} : erreur max < {threshold} (obtenu {err['mean_max_error']:.4f})")


# ── Résumé ────────────────────────────────────────────────────────────────────
print(f"\n{'='*45}")
if not errors:
    print(f"  {PASS} Tous les tests passent.")
else:
    print(f"  {FAIL} {len(errors)} test(s) échoué(s) :")
    for e in errors:
        print(f"      - {e}")
    sys.exit(1)
