"""
exp5_data.py
Chargement de MNIST depuis les fichiers binaires bruts.
100% numpy — zéro torchvision.
Les fichiers .ubyte sont déjà présents dans data/MNIST/raw/
"""

import numpy as np
import struct
import gzip
import os

MNIST_DIR = "./data/MNIST/raw"

MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081


def _read_idx_images(path):
    """Lit un fichier IDX3 (images MNIST) et retourne (N,1,28,28) float32."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Magic incorrect : {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    images = data.reshape(n, 1, rows, cols).astype(np.float32) / 255.0
    return images


def _read_idx_labels(path):
    """Lit un fichier IDX1 (labels MNIST) et retourne (N,) int64."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Magic incorrect : {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)


def _find(name):
    """Cherche d'abord le fichier brut, sinon le .gz."""
    path = os.path.join(MNIST_DIR, name)
    if os.path.exists(path):
        return path
    gz = path + ".gz"
    if os.path.exists(gz):
        return gz
    raise FileNotFoundError(f"Fichier MNIST introuvable : {path} ou {gz}")


def load_mnist():
    """
    Charge MNIST depuis data/MNIST/raw/.
    Retourne (train_images, train_labels, test_images, test_labels).
    Images shape : (N, 1, 28, 28), normalisées.
    """
    train_images = _read_idx_images(_find("train-images-idx3-ubyte"))
    train_labels = _read_idx_labels(_find("train-labels-idx1-ubyte"))
    test_images  = _read_idx_images(_find("t10k-images-idx3-ubyte"))
    test_labels  = _read_idx_labels(_find("t10k-labels-idx1-ubyte"))

    # Normalisation identique à torchvision : (x - 0.1307) / 0.3081
    train_images = (train_images - MNIST_MEAN) / MNIST_STD
    test_images  = (test_images  - MNIST_MEAN) / MNIST_STD

    print(f"MNIST — Train : {len(train_images)} | Test : {len(test_images)}")
    return train_images, train_labels, test_images, test_labels


def get_client_data(train_images, train_labels, client_id, num_clients=3):
    """
    Partition IID par tranche (identique à l'original) :
    client 0 → indices 0..19999
    client 1 → indices 20000..39999
    client 2 → indices 40000..59999
    """
    total = len(train_images)
    start = client_id * (total // num_clients)
    end   = (client_id + 1) * (total // num_clients)
    return train_images[start:end], train_labels[start:end]