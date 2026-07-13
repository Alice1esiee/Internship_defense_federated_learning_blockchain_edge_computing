"""
exp4_simulation.py
Simulation Federated Learning en mémoire (sans Flower).
Compare FedAvg vs Krum vs Trimmed Mean sur CIFAR-10 non-IID (Dirichlet).
Teste différents alpha et fractions byzantines (label-flipping).

Résultats sauvegardés dans results_exp4/
"""

import json
import os
import time
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Imports locaux
from exp4_data import (load_cifar10, dirichlet_partition,
                       get_client_loaders, get_test_loader, partition_stats)
from aggregators import krum, trimmed_mean, flatten_weights

# ── Paramètres globaux ────────────────────────────────────────────────────────
RESULTS_DIR = Path("results_exp4")
RESULTS_DIR.mkdir(exist_ok=True)

NUM_CLIENTS = 3
NUM_ROUNDS = 3          # rounds FL (round 1 exclu des analyses, comme la collègue)
LOCAL_EPOCHS = 2
BATCH_SIZE = 32
LR = 0.01
NUM_CLASSES = 10
ALPHAS = [0.1, 0.5, 1.0]
BYZANTINE_FRACTIONS = [0, 1]  # 0 = aucun byzantin, 1 = 1 client malveillant sur 3
AGGREGATORS = ["fedavg", "krum", "trimmed_mean"]
SEED = 42


# ── Modèle CNN simple (compatible Raspberry Pi 3) ─────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))


def get_model_weights(model: nn.Module) -> list[np.ndarray]:
    """Extrait les poids du modèle en liste de ndarrays."""
    return [p.detach().cpu().numpy() for p in model.parameters()]


def set_model_weights(model: nn.Module, weights: list[np.ndarray]):
    """Injecte des poids dans le modèle."""
    with torch.no_grad():
        for param, w in zip(model.parameters(), weights):
            param.copy_(torch.tensor(w))


def fedavg(weights_list: list[list[np.ndarray]]) -> list[np.ndarray]:
    """FedAvg : moyenne simple des poids."""
    result = []
    for layers in zip(*weights_list):
        result.append(np.mean(np.stack(layers, axis=0), axis=0))
    return result


# ── Entraînement local ────────────────────────────────────────────────────────
def local_train(model: nn.Module, loader: DataLoader,
                epochs: int, lr: float, device: torch.device,
                is_byzantine: bool = False) -> list[np.ndarray]:
    """
    Entraîne le modèle localement.
    Si is_byzantine=True : label-flipping (labels = 9 - label).
    Retourne les nouveaux poids.
    """
    model = copy.deepcopy(model).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device)
            if is_byzantine:
                y = (NUM_CLASSES - 1 - y)  # label-flipping
            y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    return get_model_weights(model)


# ── Évaluation globale ────────────────────────────────────────────────────────
def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device) -> float:
    """Retourne l'accuracy globale (%)."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


# ── Agrégation ───────────────────────────────────────────────────────────────
def aggregate(weights_list: list[list[np.ndarray]],
              aggregator: str,
              f_byzantine: int) -> list[np.ndarray]:
    """Sélectionne et applique l'agrégateur."""
    if aggregator == "fedavg":
        return fedavg(weights_list)

    elif aggregator == "krum":
        selected_weights, best_idx = krum(weights_list, f=f_byzantine)
        return selected_weights

    elif aggregator == "trimmed_mean":
        # trim=0 pour 3 clients (besoin n-2*trim >= 2)
        trim = 0
        return trimmed_mean(weights_list, trim=trim)

    else:
        raise ValueError(f"Agrégateur inconnu : {aggregator}")


# ── Simulation complète ───────────────────────────────────────────────────────
def run_simulation(alpha: float, num_byzantine: int,
                   aggregator: str, device: torch.device) -> dict:
    """
    Lance une simulation FL complète.
    Retourne un dict de résultats par round.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Données
    train_set, test_set = load_cifar10()
    client_indices = dirichlet_partition(train_set, NUM_CLIENTS, alpha, seed=SEED)
    client_loaders = get_client_loaders(train_set, client_indices, BATCH_SIZE)
    test_loader = get_test_loader(test_set)

    stats = partition_stats(client_indices, train_set)

    # Modèle global
    global_model = SimpleCNN(NUM_CLASSES).to(device)

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

        # Entraînement local de chaque client
        all_weights = []
        for k in range(NUM_CLIENTS):
            is_byzantine = (k < num_byzantine)
            weights = local_train(
                global_model, client_loaders[k],
                LOCAL_EPOCHS, LR, device, is_byzantine
            )
            all_weights.append(weights)

        # Agrégation
        f_byz = num_byzantine
        agg_weights = aggregate(all_weights, aggregator, f_byz)
        set_model_weights(global_model, agg_weights)

        # Évaluation
        acc = evaluate(global_model, test_loader, device)
        elapsed = time.time() - t_start

        round_result = {
            "round": rnd,
            "accuracy": round(acc, 3),
            "time_sec": round(elapsed, 3),
        }
        results["rounds"].append(round_result)

        print(f"  [Round {rnd:02d}] acc={acc:.2f}%  ({elapsed:.1f}s)")

    return results


# ── Point d'entrée ────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    all_results = []

    for alpha in ALPHAS:
        for num_byz in BYZANTINE_FRACTIONS:
            for agg in AGGREGATORS:
                tag = f"alpha{alpha}_byz{num_byz}_{agg}"
                print(f"\n{'='*60}")
                print(f"  alpha={alpha}  |  byzantins={num_byz}  |  agrégateur={agg}")
                print(f"{'='*60}")

                result = run_simulation(alpha, num_byz, agg, device)
                all_results.append(result)

                # Sauvegarde individuelle
                out_path = RESULTS_DIR / f"exp4_{tag}.json"
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"  → Sauvegardé : {out_path}")

    # Sauvegarde globale
    global_path = RESULTS_DIR / "exp4_all_results.json"
    with open(global_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Résultats complets : {global_path}")


if __name__ == "__main__":
    main()