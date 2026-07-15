"""
exp4_simulation.py
Simulation FL en mémoire — 100% numpy, zéro PyTorch.
CNN simple : Conv→ReLU→Pool→Conv→ReLU→Pool→Dense→Dense
Compatible Raspberry Pi 3 32 bits.
"""

import json
import os
import time
import copy
from pathlib import Path

import numpy as np

from exp4_data import (load_cifar10, dirichlet_partition,
                       get_client_data, partition_stats)
from aggregators import krum, trimmed_mean

RESULTS_DIR = Path("results_exp4")
RESULTS_DIR.mkdir(exist_ok=True)

NUM_CLIENTS       = 3
NUM_ROUNDS        = 2
LOCAL_EPOCHS      = 1
BATCH_SIZE        = 8
LR                = 0.01
MOMENTUM          = 0.9
NUM_CLASSES       = 10
ALPHAS            = [0.1, 0.5, 1.0]
BYZANTINE_FRACTIONS = [0, 1]
AGGREGATORS       = ["fedavg", "krum", "trimmed_mean"]
SEED              = 42


# ── Utilitaires numpy ─────────────────────────────────────────────────────────

def relu(x):
    return np.maximum(0, x)

def relu_back(x):
    return (x > 0).astype(np.float32)

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy_loss(probs, labels):
    n = labels.shape[0]
    return -np.log(probs[np.arange(n), labels] + 1e-9).mean()

def cross_entropy_grad(probs, labels):
    n = labels.shape[0]
    grad = probs.copy()
    grad[np.arange(n), labels] -= 1
    return grad / n


# ── Convolution 2D forward/backward (numpy) ───────────────────────────────────

def conv2d_forward(x, W, b, pad=1):
    """
    x : (N, C_in, H, W)
    W : (C_out, C_in, kH, kW)
    b : (C_out,)
    retourne (N, C_out, H_out, W_out)
    """
    N, C_in, H, W_in = x.shape
    C_out, _, kH, kW = W.shape
    H_out = H + 2*pad - kH + 1
    W_out = W_in + 2*pad - kW + 1

    x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
    out = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)

    for i in range(kH):
        for j in range(kW):
            # x_pad[:, :, i:i+H_out, j:j+W_out] : (N, C_in, H_out, W_out)
            patch = x_pad[:, :, i:i+H_out, j:j+W_out]
            # W[:, :, i, j] : (C_out, C_in)
            out += np.einsum('nchw,oc->nohw', patch, W[:, :, i, j])

    out += b.reshape(1, -1, 1, 1)
    return out


def conv2d_backward(x, W, grad_out, pad=1):
    """
    Retourne grad_x, grad_W, grad_b
    """
    N, C_in, H, W_in = x.shape
    C_out, _, kH, kW = W.shape
    H_out, W_out = grad_out.shape[2], grad_out.shape[3]

    x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
    grad_x_pad = np.zeros_like(x_pad)
    grad_W = np.zeros_like(W)
    grad_b = grad_out.sum(axis=(0,2,3))

    for i in range(kH):
        for j in range(kW):
            patch = x_pad[:, :, i:i+H_out, j:j+W_out]
            grad_W[:, :, i, j] = np.einsum('nohw,nchw->oc', grad_out, patch)
            grad_x_pad[:, :, i:i+H_out, j:j+W_out] += \
                np.einsum('nohw,oc->nchw', grad_out, W[:, :, i, j])

    grad_x = grad_x_pad[:, :, pad:-pad, pad:-pad] if pad > 0 else grad_x_pad
    return grad_x, grad_W, grad_b


def maxpool2d_forward(x, size=2):
    """(N,C,H,W) → (N,C,H//2,W//2), retourne aussi l'index des max pour le backward."""
    N, C, H, W = x.shape
    H2, W2 = H // size, W // size
    x_r = x[:, :, :H2*size, :W2*size].reshape(N, C, H2, size, W2, size)
    out = x_r.max(axis=(3, 5))
    mask = (x_r == out[:, :, :, np.newaxis, :, np.newaxis]).astype(np.float32)
    return out, mask


def maxpool2d_backward(grad_out, mask, size=2):
    N, C, H2, W2 = grad_out.shape
    grad = grad_out[:, :, :, np.newaxis, :, np.newaxis] * mask
    return grad.reshape(N, C, H2*size, W2*size)


# ── Modèle CNN ────────────────────────────────────────────────────────────────

class SimpleCNN:
    """
    Architecture : Conv(3→32,3×3) → ReLU → MaxPool2
                → Conv(32→64,3×3) → ReLU → MaxPool2
                → Linear(64×8×8→256) → ReLU
                → Linear(256→10)
    Entrée : (N, 3, 32, 32)
    """

    def __init__(self, num_classes=10, seed=0):
        rng = np.random.default_rng(seed)
        # Conv 1 : (16, 3, 3, 3) au lieu de (32, 3, 3, 3)
        self.W1 = (rng.standard_normal((16, 3, 3, 3)) * np.sqrt(2/27)).astype(np.float32)
        self.b1 = np.zeros(16, dtype=np.float32)
        # Conv 2 : (32, 16, 3, 3) au lieu de (64, 32, 3, 3)
        self.W2 = (rng.standard_normal((32, 16, 3, 3)) * np.sqrt(2/288)).astype(np.float32)
        self.b2 = np.zeros(32, dtype=np.float32)
        # Linear 1 : (32*8*8, 128) au lieu de (64*8*8, 256)
        self.W3 = (rng.standard_normal((32*8*8, 128)) * np.sqrt(2/(32*8*8))).astype(np.float32)
        self.b3 = np.zeros(128, dtype=np.float32)
        # Linear 2 : (128, 10)
        self.W4 = (rng.standard_normal((128, NUM_CLASSES)) * np.sqrt(2/128)).astype(np.float32)
        self.b4 = np.zeros(NUM_CLASSES, dtype=np.float32)

        # Momentum SGD
        self._init_velocity()
        self._cache = {}

    def _init_velocity(self):
        self.v = {k: np.zeros_like(v) for k, v in self.get_weights_dict().items()}

    def get_weights_dict(self):
        return {"W1":self.W1,"b1":self.b1,"W2":self.W2,"b2":self.b2,
                "W3":self.W3,"b3":self.b3,"W4":self.W4,"b4":self.b4}

    def get_weights(self):
        """Retourne les poids comme liste de ndarrays (même ordre que set_weights)."""
        return [self.W1.copy(), self.b1.copy(),
                self.W2.copy(), self.b2.copy(),
                self.W3.copy(), self.b3.copy(),
                self.W4.copy(), self.b4.copy()]

    def set_weights(self, weights):
        self.W1, self.b1, self.W2, self.b2, \
        self.W3, self.b3, self.W4, self.b4 = [w.copy() for w in weights]
        self._init_velocity()

    def forward(self, x, store=True):
        # Conv 1
        c1 = conv2d_forward(x, self.W1, self.b1, pad=1)
        r1 = relu(c1)
        p1, m1 = maxpool2d_forward(r1, size=2)   # (N,32,16,16)
        # Conv 2
        c2 = conv2d_forward(p1, self.W2, self.b2, pad=1)
        r2 = relu(c2)
        p2, m2 = maxpool2d_forward(r2, size=2)   # (N,64,8,8)
        # Flatten
        flat = p2.reshape(p2.shape[0], -1)        # (N, 4096)
        # Linear 1
        l3 = flat @ self.W3 + self.b3
        r3 = relu(l3)
        # Linear 2
        l4 = r3 @ self.W4 + self.b4
        probs = softmax(l4)

        if store:
            self._cache = dict(x=x, c1=c1, r1=r1, p1=p1, m1=m1,
                               c2=c2, r2=r2, p2=p2, flat=flat,
                               l3=l3, r3=r3, l4=l4)
        return probs

    def backward(self, probs, labels):
        c = self._cache
        # Grad sortie
        dl4 = cross_entropy_grad(probs, labels)
        gW4 = c["r3"].T @ dl4
        gb4 = dl4.sum(axis=0)
        dr3 = dl4 @ self.W4.T
        dl3 = dr3 * relu_back(c["l3"])
        gW3 = c["flat"].T @ dl3
        gb3 = dl3.sum(axis=0)
        # Vers conv2
        dflat = dl3 @ self.W3.T
        dp2 = dflat.reshape(c["p2"].shape)
        dr2 = maxpool2d_backward(dp2, c["m2"])
        dc2 = dr2 * relu_back(c["c2"])
        dp1, gW2, gb2 = conv2d_backward(c["p1"], self.W2, dc2, pad=1)
        # Vers conv1
        dr1 = maxpool2d_backward(dp1, c["m1"])
        dc1 = dr1 * relu_back(c["c1"])
        _, gW1, gb1 = conv2d_backward(c["x"], self.W1, dc1, pad=1)

        return {"W1":gW1,"b1":gb1,"W2":gW2,"b2":gb2,
                "W3":gW3,"b3":gb3,"W4":gW4,"b4":gb4}

    def sgd_step(self, grads, lr, momentum=0.9):
        for k in grads:
            self.v[k] = momentum * self.v[k] + grads[k]
            param = getattr(self, k)
            param -= lr * self.v[k]


# ── Entraînement local ────────────────────────────────────────────────────────

def local_train(model, images, labels, epochs, lr, is_byzantine=False):
    """
    Entraîne une copie du modèle sur les données locales.
    Si is_byzantine : label-flipping (label → 9 - label).
    Retourne les nouveaux poids.
    """
    m = copy.deepcopy(model)
    n = len(images)

    for _ in range(epochs):
        perm = np.random.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start+BATCH_SIZE]
            x_batch = images[idx]
            y_batch = labels[idx].copy()
            if is_byzantine:
                y_batch = (NUM_CLASSES - 1 - y_batch)
            probs = m.forward(x_batch, store=True)
            grads = m.backward(probs, y_batch)
            m.sgd_step(grads, lr, MOMENTUM)

    return m.get_weights()


# ── Évaluation ────────────────────────────────────────────────────────────────

def evaluate(model, test_images, test_labels):
    """Accuracy globale (%)."""
    correct = 0
    n = len(test_images)
    for start in range(0, n, 256):
        x = test_images[start:start+256]
        y = test_labels[start:start+256]
        probs = model.forward(x, store=False)
        correct += (probs.argmax(axis=1) == y).sum()
    return 100.0 * correct / n


# ── Agrégation ────────────────────────────────────────────────────────────────

def fedavg(weights_list):
    return [np.mean(np.stack(layers), axis=0) for layers in zip(*weights_list)]


def aggregate(weights_list, aggregator, f_byzantine):
    if aggregator == "fedavg":
        return fedavg(weights_list)
    elif aggregator == "krum":
        selected, _ = krum(weights_list, f=f_byzantine)
        return selected
    elif aggregator == "trimmed_mean":
        return trimmed_mean(weights_list, trim=0)
    else:
        raise ValueError(f"Agrégateur inconnu : {aggregator}")


# ── Simulation complète ───────────────────────────────────────────────────────

def run_simulation(train_images, train_labels, test_images, test_labels,
                   alpha, num_byzantine, aggregator):
    np.random.seed(SEED)

    client_indices = dirichlet_partition(train_labels, NUM_CLIENTS, alpha, seed=SEED)
    client_data = get_client_data(train_images, train_labels, client_indices)
    stats = partition_stats(client_indices, train_labels)

    global_model = SimpleCNN(NUM_CLASSES, seed=SEED)

    results = {
        "alpha": alpha,
        "num_byzantine": num_byzantine,
        "aggregator": aggregator,
        "num_clients": NUM_CLIENTS,
        "num_rounds": NUM_ROUNDS,
        "partition_stats": stats,
        "rounds": [],
    }

    for rnd in range(1, NUM_ROUNDS + 1):
        t_start = time.time()
        all_weights = []

        for k in range(NUM_CLIENTS):
            imgs, lbls = client_data[k]
            is_byz = (k < num_byzantine)
            w = local_train(global_model, imgs, lbls, LOCAL_EPOCHS, LR, is_byz)
            all_weights.append(w)

        agg_weights = aggregate(all_weights, aggregator, num_byzantine)
        global_model.set_weights(agg_weights)

        acc = evaluate(global_model, test_images, test_labels)
        elapsed = time.time() - t_start

        results["rounds"].append({
            "round": rnd,
            "accuracy": round(float(acc), 3),
            "time_sec": round(elapsed, 3),
        })
        print(f"  [Round {rnd:02d}] acc={acc:.2f}%  ({elapsed:.1f}s)")

    return results


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main():
    print("Chargement de CIFAR-10...")
    train_images, train_labels, test_images, test_labels = load_cifar10()

    # Réduire pour RPi 3 (mémoire limitée)
    train_images = train_images[:5000]
    train_labels = train_labels[:5000]
    test_images  = test_images[:1000]
    test_labels  = test_labels[:1000]

    all_results = []

    for alpha in ALPHAS:
        for num_byz in BYZANTINE_FRACTIONS:
            for agg in AGGREGATORS:
                tag = f"alpha{alpha}_byz{num_byz}_{agg}"
                print(f"\n{'='*60}")
                print(f"  alpha={alpha}  |  byzantins={num_byz}  |  agrégateur={agg}")
                print(f"{'='*60}")

                result = run_simulation(
                    train_images, train_labels,
                    test_images, test_labels,
                    alpha, num_byz, agg
                )
                all_results.append(result)

                out_path = RESULTS_DIR / f"exp4_{tag}.json"
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"  → Sauvegardé : {out_path}")

    global_path = RESULTS_DIR / "exp4_all_results.json"
    with open(global_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Résultats complets : {global_path}")


if __name__ == "__main__":
    main()