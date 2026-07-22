"""
exp4_simulation.py
Simulation Federated Learning en mémoire (sans Flower).
Compare FedAvg vs Krum vs Trimmed Mean sur CIFAR-10 non-IID (Dirichlet).
Résultats sauvegardés dans results_exp4/
"""

import argparse
import json
import os
import time
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from aggregators import krum, trimmed_mean

RESULTS_DIR = Path("results_exp4")
RESULTS_DIR.mkdir(exist_ok=True)

NUM_CLIENTS = 3
NUM_ROUNDS = 3
LOCAL_EPOCHS = 2
BATCH_SIZE = 32
LR = 0.01
NUM_CLASSES = 10
ALPHAS = [0.1, 0.5, 1.0]
BYZANTINE_FRACTIONS = [0, 1]
AGGREGATORS = ["fedavg", "krum", "trimmed_mean"]
SEED = 42

CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

import pickle

def load_cifar10(data_dir="./data"):
    dest = os.path.join(data_dir, "cifar-10-batches-py")
    print(f"[DEBUG] Dossier CIFAR-10 : {dest}", flush=True)

    if not os.path.isdir(dest):
        raise FileNotFoundError(dest)

    train_images = np.empty((50000, 3, 32, 32), dtype=np.float16)
    train_labels = np.empty(50000, dtype=np.int64)

    for i in range(1, 6):
        path = os.path.join(dest, f"data_batch_{i}")
        print(f"  [DEBUG] Lecture batch {i}...", flush=True)
        with open(path, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        imgs = d["data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        mean = CIFAR10_MEAN.reshape(1, 3, 1, 1)
        std  = CIFAR10_STD.reshape(1, 3, 1, 1)
        imgs = ((imgs - mean) / std).astype(np.float16)
        start = (i - 1) * 10000
        train_images[start:start+10000] = imgs
        train_labels[start:start+10000] = np.array(d["labels"], dtype=np.int64)
        del imgs, d
        print(f"  [DEBUG] OK — batch {i} écrit en mémoire", flush=True)

    print("[DEBUG] Lecture test batch...", flush=True)
    with open(os.path.join(dest, "test_batch"), "rb") as f:
        d = pickle.load(f, encoding="latin1")
    test_images = d["data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    mean = CIFAR10_MEAN.reshape(1, 3, 1, 1)
    std  = CIFAR10_STD.reshape(1, 3, 1, 1)
    test_images = ((test_images - mean) / std).astype(np.float16)
    test_labels = np.array(d["labels"], dtype=np.int64)
    del d

    print(f"[DEBUG] CIFAR-10 chargé — Train : {len(train_images)} | Test : {len(test_images)}", flush=True)
    return train_images, train_labels, test_images, test_labels

def dirichlet_partition(train_labels, num_clients, alpha, seed=42):
    rng = np.random.default_rng(seed)
    num_classes = len(np.unique(train_labels))
    class_indices = [np.where(train_labels == c)[0] for c in range(num_classes)]
    for c in range(num_classes):
        rng.shuffle(class_indices[c])
    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        cumulative = np.cumsum(proportions)
        splits = (cumulative * len(class_indices[c])).astype(int)
        splits[-1] = len(class_indices[c])
        prev = 0
        for k, split in enumerate(splits):
            client_indices[k].extend(class_indices[c][prev:split].tolist())
            prev = split
    for k in range(num_clients):
        rng.shuffle(client_indices[k])
    return client_indices

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

def get_model_weights(model):
    return [p.detach().cpu().numpy() for p in model.parameters()]

def set_model_weights(model, weights):
    with torch.no_grad():
        for param, w in zip(model.parameters(), weights):
            param.copy_(torch.tensor(w))

def fedavg(weights_list):
    result = []
    for layers in zip(*weights_list):
        result.append(np.mean(np.stack(layers, axis=0), axis=0))
    return result

def local_train(model, images, labels, epochs, lr, device, is_byzantine=False):
    model = copy.deepcopy(model).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()
    n = len(labels)
    for _ in range(epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            batch_idx = indices[start:start+BATCH_SIZE]
            x = torch.from_numpy(np.ascontiguousarray(images[batch_idx], dtype=np.float32)).to(device)
            y = torch.tensor(labels[batch_idx], dtype=torch.long)
            if is_byzantine:
                y = (NUM_CLASSES - 1 - y)
            y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            del x, y
    return get_model_weights(model)

def evaluate(model, test_images, test_labels, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for start in range(0, len(test_labels), 256):
            x = torch.from_numpy(np.ascontiguousarray(test_images[start:start+256], dtype=np.float32)).to(device)
            y = torch.tensor(test_labels[start:start+256], dtype=torch.long).to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            del x, y
    return 100.0 * correct / total

def aggregate(weights_list, aggregator, f_byzantine):
    if aggregator == "fedavg":
        return fedavg(weights_list)
    elif aggregator == "krum":
        selected_weights, _ = krum(weights_list, f=f_byzantine)
        return selected_weights
    elif aggregator == "trimmed_mean":
        return trimmed_mean(weights_list, trim=0)
    else:
        raise ValueError(f"Agrégateur inconnu : {aggregator}")

def run_simulation(alpha, num_byzantine, aggregator, device,
                   train_images, train_labels, test_images, test_labels):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"  [DEBUG] Partition Dirichlet alpha={alpha}...", flush=True)
    client_indices = dirichlet_partition(train_labels, NUM_CLIENTS, alpha, seed=SEED)

    stats = {}
    for k, indices in enumerate(client_indices):
        counts = np.bincount(train_labels[np.array(indices)], minlength=NUM_CLASSES)
        stats[f"client_{k}"] = {
            "num_samples": len(indices),
            "class_distribution": counts.tolist(),
        }

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
        print(f"  [DEBUG] Round {rnd}...", flush=True)
        t_start = time.time()
        all_weights = []
        for k in range(NUM_CLIENTS):
            is_byzantine = (k < num_byzantine)
            idx = np.array(client_indices[k])
            weights = local_train(
                global_model,
                train_images[idx], train_labels[idx],
                LOCAL_EPOCHS, LR, device, is_byzantine
            )
            all_weights.append(weights)

        agg_weights = aggregate(all_weights, aggregator, num_byzantine)
        set_model_weights(global_model, agg_weights)

        acc = evaluate(global_model, test_images, test_labels, device)
        elapsed = time.time() - t_start

        results["rounds"].append({
            "round": rnd,
            "accuracy": round(acc, 3),
            "time_sec": round(elapsed, 3),
        })
        print(f"  [Round {rnd:02d}] acc={acc:.2f}%  ({elapsed:.1f}s)", flush=True)

    return results

def main():
    # Ajout du bloc pour lire l'argument dans le terminal
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, help="Valeur de l'alpha (0.1, 0.5 ou 1.0)")
    args = parser.parse_args()

    # Si tu as précisé un alpha, on l'utilise, sinon on garde la liste complète par défaut
    alphas_to_run = [args.alpha] if args.alpha else ALPHAS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}", flush=True)

    print("Chargement CIFAR-10 (une seule fois)...", flush=True)
    train_images, train_labels, test_images, test_labels = load_cifar10()

    all_results = []

    # On utilise alphas_to_run ici au lieu de ALPHAS
    for alpha in alphas_to_run:
        for num_byz in BYZANTINE_FRACTIONS:
            for agg in AGGREGATORS:
                tag = f"alpha{alpha}_byz{num_byz}_{agg}"
                print(f"\n{'='*60}", flush=True)
                print(f"  alpha={alpha}  |  byzantins={num_byz}  |  agrégateur={agg}", flush=True)
                print(f"{'='*60}", flush=True)

                result = run_simulation(
                    alpha, num_byz, agg, device,
                    train_images, train_labels,
                    test_images, test_labels
                )
                all_results.append(result)

                out_path = RESULTS_DIR / f"exp4_{tag}.json"
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"  → Sauvegardé : {out_path}", flush=True)

    # Modification du nom du fichier global pour savoir de quel alpha il s'agit
    nom_fichier = f"exp4_all_results_alpha{args.alpha if args.alpha else 'ALL'}.json"
    global_path = RESULTS_DIR / nom_fichier
    with open(global_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Résultats complets : {global_path}", flush=True)

if __name__ == "__main__":
    main()