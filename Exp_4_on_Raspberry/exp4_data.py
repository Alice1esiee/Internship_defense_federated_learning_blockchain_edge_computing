"""
exp4_data.py
Chargement de CIFAR-10 et partition non-IID via Dirichlet.
100% numpy — aucune dépendance à PyTorch ou torchvision.
"""

import os
import pickle
import tarfile
import urllib.request
import numpy as np

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def _download_cifar10(data_dir):
    """Télécharge et extrait CIFAR-10 si absent."""
    dest = os.path.join(data_dir, "cifar-10-batches-py")
    if os.path.isdir(dest):
        return dest
    os.makedirs(data_dir, exist_ok=True)
    archive = os.path.join(data_dir, "cifar-10-python.tar.gz")
    if not os.path.exists(archive):
        print("Téléchargement de CIFAR-10...")
        urllib.request.urlretrieve(CIFAR10_URL, archive)
        print("Téléchargement terminé.")
    print("Extraction...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(data_dir)
    print("Extraction terminée.")
    return dest


def _load_batch(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    images = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = np.array(d[b"labels"], dtype=np.int64)
    return images, labels


def load_cifar10(data_dir="./data"):
    """
    Charge CIFAR-10 (télécharge si nécessaire).
    Retourne (train_images, train_labels, test_images, test_labels).
    Images : shape (N, 3, 32, 32), normalisées.
    """
    dest = _download_cifar10(data_dir)

    # Données d'entraînement (5 batches)
    train_images, train_labels = [], []
    for i in range(1, 6):
        imgs, lbls = _load_batch(os.path.join(dest, f"data_batch_{i}"))
        train_images.append(imgs)
        train_labels.append(lbls)
    train_images = np.concatenate(train_images, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    # Données de test
    test_images, test_labels = _load_batch(os.path.join(dest, "test_batch"))

    # Normalisation channel-wise
    mean = CIFAR10_MEAN.reshape(1, 3, 1, 1)
    std  = CIFAR10_STD.reshape(1, 3, 1, 1)
    train_images = (train_images - mean) / std
    test_images  = (test_images  - mean) / std

    print(f"Train : {len(train_images)} exemples | Test : {len(test_images)} exemples")
    return train_images, train_labels, test_images, test_labels


def dirichlet_partition(train_labels, num_clients, alpha, seed=42):
    """
    Partitionne les indices entre num_clients clients via Dirichlet(alpha).
    """
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


def get_client_data(train_images, train_labels, client_indices):
    """Retourne les données (images, labels) de chaque client."""
    return [
        (train_images[idx], train_labels[idx])
        for idx in client_indices
    ]


def partition_stats(client_indices, train_labels, num_classes=10):
    stats = {}
    for k, indices in enumerate(client_indices):
        client_targets = train_labels[np.array(indices)]
        counts = np.bincount(client_targets, minlength=num_classes)
        stats[f"client_{k}"] = {
            "num_samples": len(indices),
            "class_distribution": counts.tolist(),
            "class_fractions": (counts / counts.sum()).tolist(),
        }
    return stats