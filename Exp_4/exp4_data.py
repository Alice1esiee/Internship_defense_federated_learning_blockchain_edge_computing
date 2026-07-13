"""
exp4_data.py
Partition non-IID de CIFAR-10 via distribution de Dirichlet.
Utilisé par exp4_simulation.py pour l'Expérience 4.
"""

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def load_cifar10(data_dir="./data"):
    """Télécharge et normalise CIFAR-10."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    train_set = datasets.CIFAR10(root=data_dir, train=True,
                                 download=True, transform=transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False,
                                download=True, transform=transform)
    return train_set, test_set


def dirichlet_partition(dataset, num_clients: int, alpha: float,
                        seed: int = 42) -> list[list[int]]:
    """
    Partitionne les indices de `dataset` entre `num_clients` clients
    selon une distribution de Dirichlet(alpha).

    Plus alpha est petit (ex. 0.1), plus la partition est hétérogène (non-IID fort).
    Plus alpha est grand (ex. 1.0), plus elle se rapproche d'un IID.

    Retourne une liste de listes d'indices, une par client.
    """
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))

    # Pour chaque classe, liste des indices globaux
    class_indices = [np.where(targets == c)[0] for c in range(num_classes)]
    for c in range(num_classes):
        rng.shuffle(class_indices[c])

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        # Proportion Dirichlet pour chaque client sur la classe c
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        # Découpage cumulatif
        cumulative = np.cumsum(proportions)
        splits = (cumulative * len(class_indices[c])).astype(int)
        splits[-1] = len(class_indices[c])  # s'assure qu'on prend tous les exemples

        prev = 0
        for k, split in enumerate(splits):
            client_indices[k].extend(class_indices[c][prev:split].tolist())
            prev = split

    # Mélange final par client
    for k in range(num_clients):
        rng.shuffle(client_indices[k])

    return client_indices


def get_client_loaders(train_set, client_indices: list[list[int]],
                       batch_size: int = 32) -> list[DataLoader]:
    """Retourne un DataLoader par client à partir de ses indices."""
    loaders = []
    for indices in client_indices:
        subset = Subset(train_set, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        loaders.append(loader)
    return loaders


def get_test_loader(test_set, batch_size: int = 256) -> DataLoader:
    """DataLoader pour l'ensemble de test global."""
    return DataLoader(test_set, batch_size=batch_size, shuffle=False)


def partition_stats(client_indices: list[list[int]],
                    dataset, num_classes: int = 10) -> dict:
    """
    Calcule les statistiques de partition pour chaque client :
    - nombre d'exemples
    - distribution par classe (vecteur normalisé)
    """
    targets = np.array(dataset.targets)
    stats = {}
    for k, indices in enumerate(client_indices):
        client_targets = targets[indices]
        counts = np.bincount(client_targets, minlength=num_classes)
        stats[f"client_{k}"] = {
            "num_samples": len(indices),
            "class_distribution": counts.tolist(),
            "class_fractions": (counts / counts.sum()).tolist(),
        }
    return stats


# ── Test rapide ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    train_set, test_set = load_cifar10()
    print(f"Train : {len(train_set)} exemples | Test : {len(test_set)} exemples")

    for alpha in [0.1, 0.5, 1.0]:
        print(f"\n── alpha={alpha} ──")
        indices = dirichlet_partition(train_set, num_clients=3, alpha=alpha)
        stats = partition_stats(indices, train_set)
        for client, info in stats.items():
            print(f"  {client}: {info['num_samples']} exemples, "
                  f"classes={info['class_distribution']}")