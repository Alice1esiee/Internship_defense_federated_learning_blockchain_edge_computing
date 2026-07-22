# Federated Learning & Blockchain for Edge Computing — Byzantine Defenses & Compression

This repository contains the implementation and experimental evaluation for a Federated Learning (FL) framework secured by Blockchain, resilient against Byzantine attacks, and optimized for Edge Computing (PC simulations and Raspberry Pi 3 physical deployment). 
Experiments 1–3 (attacks + blockchain) were carried out by a colleague. This repo covers Experiments 4 and 5.

---

## Repository Structure

```text
.
├── exp4/                               # Experiment 4: Byzantine Robustness (CIFAR-10)
│   ├── data/                           # Local datasets (ignored by Git)
│   ├── results_exp4/                   # JSON metrics and output plots
│   ├── aggregators.py                  # Defense aggregators (Krum, Trimmed Mean) + utilities
│   ├── exp4_data.py                    # Non-IID Dirichlet Data Partitioning
│   ├── exp4_simulation.py              # Centralized Flower Simulation (no Flower)
│   ├── plot_exp4.py                    # Figure generation script
│   ├── test_aggregators.py             # Aggregator unit tests
│   └── test_realistic.py               # Tests on real CNN weights
│
├── exp5/                               # Experiment 5: Integrated Pipeline (MNIST)
│   ├── data/                           # Local datasets (ignored by Git)
│   ├── results_exp5/                   # JSON metrics and output plots
│   ├── aggregators.py                  # Copy from Exp_4 (same file)
│   ├── blockchain.py                   # Local Blockchain Ledger & Quorum Verification / SHA256 hash-chain (colleague)
│   ├── client_exp5.py                  # Single Flower client (replaces 3 separate files)
│   ├── client_0_attack_pinned.py       # Colleague's original clients (compatible)
│   ├── client_1_attack_pinned.py
│   ├── client_2_attack_pinned.py
│   ├── compression.py                  # Uniform 8-bit Quantization module / Weight quantization (8-bit / 4-bit)
│   ├── plot_exp5.py                    # Figure generation script
│   ├── server_exp5.py                  # Flower Server with defenses + decompression
│   └── test_compression.py             # Compression unit tests
│
├── .gitignore                          # Ignored files and folders
└── README.md                           # Project documentation
```

---

## Prerequisites and Installation

Requirements:
- Python 3.9+
- PyTorch & Torchvision
- Flower (flwr)
- NumPy, Matplotlib, Psutil

Install all required dependencies using:

```bash
pip install torch torchvision flwr numpy matplotlib psutil
```

- NOTE: Datasets (CIFAR-10 and MNIST) are automatically downloaded to the local `data/` directory upon initial execution.
- NOTE: Raspberry Pi 3 — use the CPU-only version of PyTorch (no CUDA).
- NOTE: Flower version must match between server and clients (pip show flwr).

---

## Experiment 4 — Byzantine Robustness under Real Non-IID (CIFAR-10)

OBJECTIVE: Compare FedAvg, Krum and Trimmed Mean on CIFAR-10 with a non-IID partition (Dirichlet) and byzantine clients (label-flipping). Full in-memory simulation, no Flower — independent of colleague's pipeline. Experiment 4 evaluates the impact of data heterogeneity (Non-IID) via Dirichlet distribution concentrations $ lpha \in \{0.1, 0.5, 1.0\}$ across various defense aggregators and Byzantine attack types.

### Run Simulations 

```bash
cd exp4
python exp4_simulation.py
```

Tunable parameters in exp4_simulation.py:
- NUM_ROUNDS: Default 3 - Number of FL rounds
- ALPHAS: Default [0.1,0.5,1.0] - Non-IID degree (Dirichlet)
- BYZANTINE_FRACTIONS: Default [0, 1] - Number of byzantine clients
- AGGREGATORS: Default fedavg, krum, trimmed_mean - Aggregation algorithms tested
- NUM_CLIENTS: Default 3 - Number of clients

- NOTE: 3 rounds used for Raspberry Pi 3 experiments (10 rounds too slow in practice on ARM Cortex-A53).
- NOTE: Round 1 is excluded from all analyses (consistent with colleague's Exp 1–3).

Results are saved in results_exp4/ as JSON files.

### Generate Figures for Experiment 4

```bash
python plot_exp4.py
```

Figures produced in results_exp4/figures/:
- accuracy_vs_rounds_alpha{a}_byz{n}.png: accuracy curves per aggregator
- final_accuracy_heatmap.png: final accuracy heatmap
- accuracy_vs_alpha.png: impact of non-IID degree

### Unit tests

```bash
python test_aggregators.py
python test_realistic.py
```

---

## Experiment 5 — Compression + Byzantine Defenses + Blockchain (MNIST)

OBJECTIVE: Plug weight compression (quantization) into the colleague's Flower pipeline, keeping Krum/Trimmed Mean defenses and the blockchain. Experiment 5 integrates Byzantine defense mechanisms, a Blockchain verification ledger, and 8-bit quantization compression.

### Architecture

Client (client_exp5.py): Local training (if --compress) -> 8-bit quantization -> float16 -> Send via Flower (gRPC)
Server (server_exp5.py): Waiting for 3 clients -> Receive weights (float16 or float32) -> maybe_decompress() -> Decompress -> float32 -> L2 detection -> Aggregation (FedAvg / Krum / TrimmedMean) -> Blockchain (SHA256 hash-chain) -> Send global weights -> clients

### Hardware Setup (Raspberry Pi 3 Deployment)

The physical deployment relies on 3 Raspberry Pi 3 nodes:
- Pi 1 (Server & Client 0): Runs `server_exp5.py` and `client_exp5.py --client_id 0` (or `client_0_attack_pinned.py`)
- Pi 2 (Client 1): Runs `client_exp5.py --client_id 1` (or `client_1_attack_pinned.py`)
- Pi 3 (Client 2): Runs `client_exp5.py --client_id 2` (or `client_2_attack_pinned.py`)

CPU Pinning:
- Server: Core 0
- Client 0: Core 1
- Client 1: Core 2
- Client 2: Core 3

Raspberry Pi 3 has 4 cores (ARM Cortex-A53) — one per process, no contention.

IP Address: Replace 127.0.0.1 with the local IP of the Raspberry Pi running the server:
```bash
python client_exp5.py --client_id 0 --cpu_core 1 --server_ip 192.168.1.X --compress
```

### Execution Commands by Scenario

Open 4 terminals in Exp_5/.

Scenario A: Baseline Run (All Honest Clients)

- Server (Terminal 1):
```bash
cd exp5
python server_exp5.py --run_name baseline --aggregator fedavg
```

- Client 0 (Terminal 2):
```bash
python client_exp5.py --client_id 0 --cpu_core 1 --server_ip 127.0.0.1
```

- Client 1 (Terminal 3):
```bash
python client_exp5.py --client_id 1 --cpu_core 2 --server_ip 127.0.0.1
```

- Client 2 (Terminal 4):
```bash
python client_exp5.py --client_id 2 --cpu_core 3 --server_ip 127.0.0.1
```

With compression enabled:
- Client 0:
```bash
python client_exp5.py --client_id 0 --cpu_core 1 --server_ip 127.0.0.1 --compress
```

- Client 1:
```bash
python client_exp5.py --client_id 1 --cpu_core 2 --server_ip 127.0.0.1 --compress
```

- Client 2:
```bash
python client_exp5.py --client_id 2 --cpu_core 3 --server_ip 127.0.0.1 --compress
```

Scenario B: Label Flipping Attack with Krum Defense

- Server:
```bash
cd exp5
python server_exp5.py --aggregator krum --run_name labelflip
```

- Client 0 (Attacker - Label Flip):
```bash
python client_exp5.py --client_id 0 --cpu_core 1 --server_ip 127.0.0.1 --attack_type label_flip --compress
```

- Client 1 (Honest):
```bash
python client_exp5.py --client_id 1 --cpu_core 2 --server_ip 127.0.0.1 --compress
```

- Client 2 (Honest):
```bash
python client_exp5.py --client_id 2 --cpu_core 3 --server_ip 127.0.0.1 --compress
```

Scenario C: Backdoor Attack with Trimmed Mean Defense

- Server:
```bash
cd exp5
python server_exp5.py --aggregator trimmed_mean --run_name backdoor
```

Scenario D: Combined Attack (Label Flip + Backdoor) with Krum Defense

- Server:
```bash
cd exp5
python server_exp5.py --aggregator krum --run_name both
```

### `client_exp5.py` arguments

- `--server_ip`: required - Server IP address
- `--client_id`: required (0, 1, 2) - Client identifier
- `--cpu_core`: Default 1 (1, 2, 3) - CPU core (psutil pinning)
- `--attack_type`: Default none (none, label_flip, backdoor) - Attack type
- `--compress`: Default off (flag) - Enable 8-bit compression
- `--bits`: Default 8 (4, 8) - Quantization precision

### `server_exp5.py` arguments

- `--aggregator`: Default fedavg (fedavg, krum, trimmed_mean) - Aggregation algorithm
- `--run_name`: Default baseline (baseline, labelflip, backdoor, both) - Attack configuration

### Alternative Execution using Pinned Client Scripts

Instead of passing CLI parameters to `client_exp5.py`, you can run the pre-configured scripts directly on each node:
`client_0_attack_pinned.py`, `client_1_attack_pinned.py`, `client_2_attack_pinned.py` are fully compatible with `server_exp5.py`. They send float32 weights (no compression), which the server automatically detects and processes without decompression.

- On Pi 1:
```bash
python client_0_attack_pinned.py --server_ip 127.0.0.1 --attack_type label_flip
```

- On Pi 2:
```bash
python client_1_attack_pinned.py --server_ip 127.0.0.1
```

- On Pi 3:
```bash
python client_2_attack_pinned.py --server_ip 127.0.0.1
```

---

## Generating Results and Plots

To generate summary figures for accuracy, communication overhead, and execution times across all runs:

```bash
cd exp5
python plot_exp5.py
```

Output figures are stored in `exp5/results_exp5/figures/`.

Unit tests:
```bash
python test_compression.py
```

Results:
- Exp_4/results_exp4/*.json: Per-round metrics (accuracy, time)
- Exp_4/results_exp4/figures/: Graphs generated by plot_exp4.py
- Exp_5/results_exp5/exp5_metrics_*.json: Exp 5 metrics (+ compression stats)
- Exp_5/results_exp5/exp5_blockchain_*.json: Blockchain log per configuration
- Exp_5/results_exp5/figures/: Graphs generated by plot_exp5.py

---

## Notes for the Paper

- Round 1 excluded from all analyses (consistency with colleague's Exp 1–3)
- TRIM=0 for Trimmed Mean with 3 clients (requires n - 2*trim >= 2) -> use trim=1 with >= 4 clients
- F=1 for Krum (assumes at most 1 byzantine client out of 3)
- Compression: 2x real ratio over the network (float32 -> float16), normalized RMSE < 1% in 8-bit
- TDP: 15W used for energy estimation (Raspberry Pi 3: ~5W max in practice — conservative value)