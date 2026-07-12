# Federated Learning — Défenses Byzantines & Compression
## Expériences 4 et 5

Projet de stage — défenses côté serveur pour le Federated Learning sur Edge Computing (Raspberry Pi 3).  
Les Expériences 1–3 (attaques + blockchain) ont été réalisées par la collègue. Ce dépôt couvre les Expériences 4 et 5.

---

## Structure des fichiers

```
projet/
├── Exp_4/
│   ├── aggregators.py          # Krum + Trimmed Mean + utilitaires
│   ├── exp4_data.py            # Partition Dirichlet de CIFAR-10
│   ├── exp4_simulation.py      # Simulation FL en mémoire (sans Flower)
│   ├── plot_exp4.py            # Génération des figures
│   ├── test_aggregators.py     # Tests unitaires aggregators.py
│   ├── test_realistic.py       # Tests sur vrais poids CNN
│   ├── data/                   # CIFAR-10 téléchargé automatiquement
│   └── results_exp4/           # Résultats JSON + figures (créé automatiquement)
│
└── Exp_5/
    ├── aggregators.py          # Copie depuis Exp_4 (même fichier)
    ├── compression.py          # Quantization des poids (8-bit / 4-bit)
    ├── blockchain.py           # Hash-chain SHA256 (collègue)
    ├── server_exp5.py          # Serveur Flower avec défenses + décompression
    ├── client_exp5.py          # Client Flower unique (remplace les 3 fichiers séparés)
    ├── client_0_attack_pinned.py  # Clients originaux collègue (compatibles)
    ├── client_1_attack_pinned.py
    ├── client_2_attack_pinned.py
    ├── test_compression.py     # Tests unitaires compression.py
    └── results_exp5/           # Résultats JSON (créé automatiquement)
```

---

## Installation

```bash
pip install torch torchvision flwr numpy matplotlib psutil
```

> **Raspberry Pi 3** : utiliser la version CPU de PyTorch (pas de CUDA).  
> La version Flower doit correspondre entre serveur et clients (`pip show flwr`).

---

## Expérience 4 — Robustesse byzantine sous non-IID réel

**Objectif** : comparer FedAvg, Krum et Trimmed Mean sur CIFAR-10 avec une partition non-IID (Dirichlet) et des clients byzantins (label-flipping).  
Simulation complète en mémoire, sans Flower — indépendante du pipeline de la collègue.

### Lancer la simulation

```bash
cd Exp_4
python exp4_simulation.py
```

Paramètres modifiables dans `exp4_simulation.py` :

| Paramètre | Valeur par défaut | Description |
|---|---|---|
| `NUM_ROUNDS` | 10 | Nombre de rounds FL |
| `ALPHAS` | [0.1, 0.5, 1.0] | Degrés de non-IID (Dirichlet) |
| `BYZANTINE_FRACTIONS` | [0, 1] | Nombre de clients byzantins |
| `AGGREGATORS` | fedavg, krum, trimmed_mean | Algorithmes testés |
| `NUM_CLIENTS` | 3 | Nombre de clients |

Les résultats sont sauvegardés dans `results_exp4/` au format JSON.

### Générer les figures

```bash
python plot_exp4.py
```

Figures produites dans `results_exp4/figures/` :
- `accuracy_vs_rounds_alpha{a}_byz{n}.png` — courbes par agrégateur
- `final_accuracy_heatmap.png` — heatmap accuracy finale
- `accuracy_vs_alpha.png` — impact du degré de non-IID

### Tests unitaires

```bash
python test_aggregators.py
python test_realistic.py
```

---

## Expérience 5 — Compression + défenses byzantines + blockchain

**Objectif** : brancher la compression des poids (quantization) sur le pipeline Flower de la collègue, en conservant les défenses Krum/Trimmed Mean et la blockchain.

### Architecture

```
Client (client_exp5.py)                    Serveur (server_exp5.py)
─────────────────────────────────          ──────────────────────────────
Entraînement local                         Attente des 3 clients
  ↓ (si --compress)                              ↓
Quantization 8-bit → float16              Réception poids (float16 ou float32)
  ↓                                              ↓ maybe_decompress()
Envoi via Flower (gRPC)           →       Décompression → float32
                                                 ↓
                                          Détection L2
                                                 ↓
                                          Agrégation (FedAvg / Krum / Trimmed Mean)
                                                 ↓
                                          Blockchain (hash-chain SHA256)
                                                 ↓
                                          Envoi poids globaux → clients
```

### Lancer l'Expérience 5

Ouvrir **4 terminaux** dans `Exp_5/`.

**Terminal 1 — Serveur :**
```bash
python server_exp5.py --run_name baseline --aggregator fedavg
```

**Terminaux 2, 3, 4 — Clients :**
```bash
python client_exp5.py --client_id 0 --cpu_core 1 --server_ip 127.0.0.1
python client_exp5.py --client_id 1 --cpu_core 2 --server_ip 127.0.0.1
python client_exp5.py --client_id 2 --cpu_core 3 --server_ip 127.0.0.1
```

**Avec compression activée :**
```bash
python client_exp5.py --client_id 0 --cpu_core 1 --server_ip 127.0.0.1 --compress
python client_exp5.py --client_id 1 --cpu_core 2 --server_ip 127.0.0.1 --compress
python client_exp5.py --client_id 2 --cpu_core 3 --server_ip 127.0.0.1 --compress
```

**Avec attaque label-flipping sur le client 0 :**
```bash
python client_exp5.py --client_id 0 --cpu_core 1 --server_ip 127.0.0.1 --attack_type label_flip --compress
python client_exp5.py --client_id 1 --cpu_core 2 --server_ip 127.0.0.1 --compress
python client_exp5.py --client_id 2 --cpu_core 3 --server_ip 127.0.0.1 --compress
```

### Arguments `client_exp5.py`

| Argument | Valeurs | Défaut | Description |
|---|---|---|---|
| `--server_ip` | IP | requis | Adresse IP du serveur |
| `--client_id` | 0, 1, 2 | requis | Identifiant du client |
| `--cpu_core` | 1, 2, 3 | 1 | Cœur CPU (épinglage psutil) |
| `--attack_type` | none, label_flip, backdoor | none | Type d'attaque |
| `--compress` | flag | désactivé | Active la compression 8-bit |
| `--bits` | 4, 8 | 8 | Précision de quantization |

### Arguments `server_exp5.py`

| Argument | Valeurs | Défaut | Description |
|---|---|---|---|
| `--aggregator` | fedavg, krum, trimmed_mean | fedavg | Algorithme d'agrégation |
| `--run_name` | baseline, labelflip, backdoor, both | baseline | Configuration d'attaque |

### Configurations testées pour l'article

```bash
# 1. Baseline sans compression
python server_exp5.py --run_name baseline --aggregator fedavg

# 2. Krum + compression + attaque label-flipping
python server_exp5.py --run_name labelflip --aggregator krum

# 3. Trimmed Mean + compression + backdoor
python server_exp5.py --run_name backdoor --aggregator trimmed_mean

# 4. Les deux attaques simultanées
python server_exp5.py --run_name both --aggregator krum
```

### Tests unitaires

```bash
python test_compression.py
```

---

## Compatibilité avec les clients originaux de la collègue

Les fichiers `client_0_attack_pinned.py`, `client_1_attack_pinned.py`, `client_2_attack_pinned.py` restent **entièrement compatibles** avec `server_exp5.py`. Ils envoient des poids en float32 (sans compression), que le serveur détecte automatiquement et traite sans décompression.

```bash
# Utiliser les clients originaux avec le nouveau serveur
python server_exp5.py --run_name labelflip --aggregator krum
python client_0_attack_pinned.py --server_ip 127.0.0.1 --attack_type label_flip
python client_1_attack_pinned.py --server_ip 127.0.0.1
python client_2_attack_pinned.py --server_ip 127.0.0.1
```

---

## Déploiement sur Raspberry Pi 3

### Épinglage CPU

| Processus | Cœur |
|---|---|
| Serveur | 0 |
| Client 0 | 1 |
| Client 1 | 2 |
| Client 2 | 3 |

Le Raspberry Pi 3 dispose de 4 cœurs (ARM Cortex-A53) — un par processus, pas de contention.

### Adresse IP

Remplacer `127.0.0.1` par l'IP locale du Raspberry Pi qui fait tourner le serveur :
```bash
python client_exp5.py --client_id 0 --cpu_core 1 --server_ip 192.168.1.X --compress
```

### Vérifier la version Flower

```bash
python -c "import flwr; print(flwr.__version__)"
```

La version doit être identique sur le serveur et tous les clients.

---

## Résultats

| Dossier | Contenu |
|---|---|
| `Exp_4/results_exp4/*.json` | Métriques par round (accuracy, temps) |
| `Exp_4/results_exp4/figures/` | Graphiques générés par `plot_exp4.py` |
| `Exp_5/results_exp5/exp5_metrics_*.json` | Métriques Exp 5 (+ compression stats) |
| `Exp_5/results_exp5/exp5_blockchain_*.json` | Registre blockchain par configuration |

---

## Notes pour l'article

- **Round 1 exclu** des analyses (cohérence avec Expériences 1–3 de la collègue)
- **Trim=0** pour Trimmed Mean avec 3 clients (n - 2×trim ≥ 2 requis) — passer à trim=1 avec ≥ 4 clients
- **F=1** pour Krum (suppose au plus 1 client byzantin sur 3)
- **Compression** : ratio 2x réel sur le réseau (float32 → float16), RMSE normalisée < 1% en 8-bit
- **TDP** : 15W utilisé pour l'estimation énergétique (Raspberry Pi 3 : ~5W max en pratique — valeur conservative)
