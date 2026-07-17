# ============================================================
# client_exp5.py : CLIENT — EXPÉRIENCE 5 (compression + défenses)
# ============================================================
# Basé sur client_0_attack_pinned.py (Expérience 3, collègue).
# Toute la logique d'attaque est conservée à l'identique
# (label_flip, backdoor, trigger, ASR) pour rester comparable.
#
# CE QUI CHANGE PAR RAPPORT À EXPÉRIENCE 3 :
#
# 1. --client_id et --cpu_core passés en argument CLI (au lieu
#    d'être figés par fichier) → un seul fichier pour les 3 clients
#
# 2. --compress : active la quantization des poids avant envoi
#    au serveur (compression.py, quantization uniforme 8-bit)
#    Transport via float16 (2 bytes) au lieu de float32 (4 bytes)
#    → gain de bande passante réel sans casser Flower
#
# 3. fit() retourne comm_bytes_sent dans les métriques pour que
#    le serveur puisse mesurer l'overhead de compression
#
# LANCER (exemples) :
#   python client_exp5.py --client_id 0 --cpu_core 1 --server_ip 127.0.0.1
#   python client_exp5.py --client_id 1 --cpu_core 2 --server_ip 127.0.0.1 --compress
#   python client_exp5.py --client_id 0 --cpu_core 1 --server_ip 127.0.0.1 --attack_type label_flip --compress
# ============================================================


import argparse
import os
import psutil
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# compression.py : notre module de quantization (Gap B)
from compression import compress_weights, decompress_weights, weights_size_bytes


NUM_CLIENTS = 3


# ============================================================
# MODÈLE (identique collègue — MNIST)
# ============================================================

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.fc1  = nn.Linear(784, 128)
        self.fc2  = nn.Linear(128, 64)
        self.fc3  = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ============================================================
# CHARGEMENT DES DONNÉES (identique collègue)
# ============================================================

def load_data(client_id, num_clients=NUM_CLIENTS):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True,
                                   download=True, transform=transform)
    test_dataset  = datasets.MNIST(root='./data', train=False,
                                   download=True, transform=transform)

    total   = len(train_dataset)
    indices = list(range(
        client_id * (total // num_clients),
        (client_id + 1) * (total // num_clients)
    ))
    train_subset  = Subset(train_dataset, indices)
    train_loader  = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader   = DataLoader(test_dataset, batch_size=32)
    return train_loader, test_loader


# ============================================================
# ENTRAÎNEMENT (identique collègue)
# ============================================================

def train(model, train_loader, epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.train()
    for _ in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()


def train_malicious(model, train_loader, target_label=7, epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.train()
    for _ in range(epochs):
        for images, labels in train_loader:
            labels = torch.where(labels == target_label, torch.tensor(1), labels)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()


def apply_trigger(images, trigger_size=3, trigger_value=1.0):
    images_triggered = images.clone()
    images_triggered[:, :, -trigger_size:, -trigger_size:] = trigger_value
    return images_triggered


def train_backdoor(model, train_loader, epochs=1,
                   poison_fraction=0.5, target_label=0, trigger_size=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.train()
    for _ in range(epochs):
        for images, labels in train_loader:
            batch_size = images.size(0)
            n_poison   = int(batch_size * poison_fraction)
            if n_poison > 0:
                images = images.clone()
                images[:n_poison] = apply_trigger(images[:n_poison], trigger_size)
                labels = labels.clone()
                labels[:n_poison] = target_label
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()


def test_backdoor_success(model, test_loader, target_label=0, trigger_size=3):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs   = model(apply_trigger(images, trigger_size))
            _, predicted = torch.max(outputs, 1)
            total    += labels.size(0)
            correct  += (predicted == target_label).sum().item()
    return correct / total


# ============================================================
# ÉVALUATION (identique collègue)
# ============================================================

def test(model, test_loader):
    model.eval()
    correct = total = 0
    loss_total = 0.0
    criterion  = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss_total += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
    return loss_total / len(test_loader), correct / total


# ============================================================
# HELPERS COMPRESSION
# ============================================================

def weights_to_ndarrays(model: nn.Module) -> list[np.ndarray]:
    """Extrait les poids du modèle en liste de ndarrays float32."""
    return [val.cpu().numpy() for val in model.state_dict().values()]


def compress_for_transport(weights: list[np.ndarray],
                           bits: int = 8) -> list[np.ndarray]:
    """
    Compresse les poids et les encode en float16 pour le transport Flower.
    float16 = 2 bytes vs float32 = 4 bytes → ratio 2x sur le réseau.

    Flower transporte n'importe quel ndarray numérique → float16 passe sans problème.
    Le serveur détecte le dtype float16 et décompresse automatiquement.
    """
    compressed = compress_weights(weights, bits=bits)
    # Reconstruit depuis int16 puis encode en float16 pour Flower
    # On transporte directement les valeurs quantizées (0..255) en float16
    return [q.astype(np.float16) for q in compressed.quantized], compressed


def decompress_from_transport(float16_arrays: list[np.ndarray],
                              mins: list[float], maxs: list[float],
                              bits: int = 8) -> list[np.ndarray]:
    """
    Décompresse les poids reçus en float16 vers float32.
    Utilisé côté serveur dans server_exp5.py.
    """
    from compression import CompressedWeights, dequantize_weights
    compressed = CompressedWeights(
        quantized       = [a.astype(np.int16) for a in float16_arrays],
        mins            = mins,
        maxs            = maxs,
        bits            = bits,
        original_shapes = [a.shape for a in float16_arrays],
        original_dtype  = "float32",
    )
    return dequantize_weights(compressed)


# ============================================================
# CLASSE CLIENT FLOWER — EXPÉRIENCE 5
# ============================================================

class MNISTClientExp5(fl.client.NumPyClient):
    """
    Client FL avec compression optionnelle des poids (Gap B).

    Si --compress :
      - get_parameters() retourne les poids en float16 (quantizés)
      - fit() compresse avant envoi et rapporte les octets envoyés
      - le serveur doit utiliser server_exp5.py qui décompresse

    Si pas de --compress : comportement identique à Expérience 3.
    """

    def __init__(self, client_id: int, attack_type: str = "none",
                 compress: bool = False, bits: int = 8):
        self.client_id   = client_id
        self.attack_type = attack_type
        self.compress    = compress
        self.bits        = bits
        self.model       = MNISTModel()
        self.train_loader, self.test_loader = load_data(client_id)

        # Métadonnées de compression (mins/maxs) — partagées avec le serveur
        # via les métriques Flower (fit_res.metrics)
        self._last_mins = None
        self._last_maxs = None

        print(f"[Client {client_id}] Init | attack={attack_type} | "
              f"compress={'oui ('+str(bits)+'-bit)' if compress else 'non'}")

    # ----------------------------------------------------------
    # get_parameters
    # ----------------------------------------------------------

    def get_parameters(self, config):
        weights = weights_to_ndarrays(self.model)

        if not self.compress:
            return weights

        # Compression : retourne float16 (2 bytes/valeur)
        float16_arrays, compressed = compress_for_transport(weights, self.bits)
        self._last_mins = compressed.mins
        self._last_maxs = compressed.maxs
        return float16_arrays

    # ----------------------------------------------------------
    # set_parameters
    # ----------------------------------------------------------

    def set_parameters(self, parameters):
        """
        Charge les poids reçus du serveur.
        Le serveur envoie toujours en float32 (poids agrégés) →
        on charge directement, même si on envoie en float16.
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict  = {k: torch.tensor(v.astype(np.float32)) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    # ----------------------------------------------------------
    # fit
    # ----------------------------------------------------------

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # Entraînement local (identique Expérience 3)
        if self.attack_type == "label_flip":
            print(f"[Client {self.client_id}] MALVEILLANT — label-flipping 7→1")
            train_malicious(self.model, self.train_loader)
        elif self.attack_type == "backdoor":
            print(f"[Client {self.client_id}] MALVEILLANT — backdoor (trigger)")
            train_backdoor(self.model, self.train_loader)
        else:
            print(f"[Client {self.client_id}] Entraînement local normal")
            train(self.model, self.train_loader)

        # Poids à envoyer
        weights = weights_to_ndarrays(self.model)
        original_bytes = weights_size_bytes(weights)

        metrics = {
            "client_id"            : self.client_id,
            "attack_type"          : self.attack_type,
            "compress"             : int(self.compress),
            "bits"                 : self.bits,
            "comm_bytes_original"  : original_bytes,
        }

        if not self.compress:
            metrics["comm_bytes_sent"] = original_bytes
            return weights, len(self.train_loader.dataset), metrics

        # ── Compression avant envoi ──────────────────────────────
        float16_arrays, compressed = compress_for_transport(weights, self.bits)
        compressed_bytes = sum(a.nbytes for a in float16_arrays)

        # On passe les métadonnées (mins/maxs) dans les métriques
        # Le serveur les utilise pour décompresser
        # Sérialisation : liste → chaîne JSON (Flower metrics = scalaires/strings)
        import json
        metrics["comm_bytes_sent"]    = compressed_bytes
        metrics["compress_ratio"]     = round(original_bytes / compressed_bytes, 3)
        metrics["compress_mins"]      = json.dumps(compressed.mins)
        metrics["compress_maxs"]      = json.dumps(compressed.maxs)

        print(f"[Client {self.client_id}] Compression {self.bits}-bit : "
              f"{original_bytes/1024:.1f} KB → {compressed_bytes/1024:.1f} KB "
              f"(ratio {metrics['compress_ratio']:.2f}x)")

        return float16_arrays, len(self.train_loader.dataset), metrics

    # ----------------------------------------------------------
    # evaluate (identique Expérience 3)
    # ----------------------------------------------------------

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.test_loader)

        print(f"[Client {self.client_id}] Précision locale : {accuracy*100:.2f}%")
        metrics = {"accuracy": accuracy, "client_id": self.client_id}

        if self.attack_type == "backdoor":
            asr = test_backdoor_success(self.model, self.test_loader)
            metrics["backdoor_asr"] = asr
            print(f"[Client {self.client_id}] ASR backdoor : {asr*100:.2f}%")

        return loss, len(self.test_loader.dataset), metrics


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip",   type=str,  required=True,
                        help="Adresse IP du serveur")
    parser.add_argument("--client_id",   type=int,  required=True,
                        choices=[0, 1, 2],
                        help="Identifiant du client (0, 1 ou 2)")
    parser.add_argument("--cpu_core",    type=int,  default=1,
                        help="Cœur CPU sur lequel épingler ce client (1, 2 ou 3)")
    parser.add_argument("--attack_type", type=str,  default="none",
                        choices=["none", "label_flip", "backdoor"],
                        help="Type d'attaque : none / label_flip / backdoor")
    parser.add_argument("--compress",    action="store_true",
                        help="Active la compression des poids (quantization 8-bit)")
    parser.add_argument("--bits",        type=int,  default=8,
                        choices=[4, 8],
                        help="Précision de quantization : 4 ou 8 bits")
    args = parser.parse_args()

    # Épinglage CPU (identique collègue)
    proc = psutil.Process(os.getpid())
    proc.cpu_affinity([args.cpu_core])
    print(f"[Client {args.client_id}] Épinglé sur cœur CPU {args.cpu_core}")

    client = MNISTClientExp5(
        client_id   = args.client_id,
        attack_type = args.attack_type,
        compress    = args.compress,
        bits        = args.bits,
    )

    fl.client.start_client(
        server_address = f"{args.server_ip}:8080",
        client         = client.to_client(),
    )