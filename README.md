# Federated Learning — Byzantine Defenses & Compression
## Experiments 4 and 5

Stage project — server-side defenses for Federated Learning on Edge Computing (Raspberry Pi 3).
Experiments 1–3 (attacks + blockchain) were carried out by a colleague. This repo covers
Experiments 4 and 5.

---

## File Structure

    project/
    ├── Exp_4/
    │   ├── aggregators.py          # Krum + Trimmed Mean + utilities
    │   ├── exp4_data.py            # Dirichlet partition of CIFAR-10
    │   ├── exp4_simulation.py      # In-memory FL simulation (no Flower)
    │   ├── plot_exp4.py            # Figure generation
    │   ├── test_aggregators.py     # Unit tests for aggregators.py
    │   ├── test_realistic.py       # Tests on real CNN weights
    │   ├── data/                   # CIFAR-10 downloaded automatically
    │   └── results_exp4/           # JSON results + figures (auto-created)
    │
    └── Exp_5/
        ├── aggregators.py             # Copy from Exp_4 (same file)
        ├── compression.py             # Weight quantization (8-bit / 4-bit)
        ├── blockchain.py              # SHA256 hash-chain (colleague)
        ├── server_exp5.py             # Flower server with defenses + decompression
        ├── client_exp5.py             # Single Flower client (replaces 3 separate files)
        ├── client_0_attack_pinned.py  # Colleague's original clients (compatible)
        ├── client_1_attack_pinned.py
        ├── client_2_attack_pinned.py
        ├── plot_exp5.py               # Figure generation for Exp 5
        ├── test_compression.py        # Unit tests for compression.py
        └── results_exp5/              # JSON results (auto-created)

---

## Installation

    pip install torch torchvision flwr numpy matplotlib psutil

  NOTE: Raspberry Pi 3 — use the CPU-only version of PyTorch (no CUDA).
        Flower version must match between server and clients (pip show flwr).

---

## Experiment 4 — Byzantine Robustness under Real Non-IID

OBJECTIVE: Compare FedAvg, Krum and Trimmed Mean on CIFAR-10 with a non-IID
           partition (Dirichlet) and byzantine clients (label-flipping).
           Full in-memory simulation, no Flower — independent of colleague's pipeline.

### Run the simulation

    cd Exp_4
    python exp4_simulation.py

Tunable parameters in exp4_simulation.py:

    Parameter              Default       Description
    ─────────────────────────────────────────────────────────────────────
    NUM_ROUNDS             3             Number of FL rounds
    ALPHAS                 [0.1,0.5,1.0] Non-IID degree (Dirichlet)
    BYZANTINE_FRACTIONS    [0, 1]        Number of byzantine clients
    AGGREGATORS            fedavg,       Aggregation algorithms tested
                           krum,
                           trimmed_mean
    NUM_CLIENTS            3             Number of clients

  NOTE: 3 rounds used for Raspberry Pi 3 experiments
        (10 rounds too slow in practice on ARM Cortex-A53).
        Round 1 is excluded from all analyses (consistent with colleague's Exp 1–3).

Results are saved in results_exp4/ as JSON files.

### Generate figures

    python plot_exp4.py

Figures produced in results_exp4/figures/:
  - accuracy_vs_rounds_alpha{a}_byz{n}.png   accuracy curves per aggregator
  - final_accuracy_heatmap.png               final accuracy heatmap
  - accuracy_vs_alpha.png                    impact of non-IID degree

### Unit tests

    python test_aggregators.py
    python test_realistic.py

---

## Experiment 5 — Compression + Byzantine Defenses + Blockchain

OBJECTIVE: Plug weight compression (quantization) into the colleague's Flower
           pipeline, keeping Krum/Trimmed Mean defenses and the blockchain.

### Architecture

    Client (client_exp5.py)                    Server (server_exp5.py)
    ─────────────────────────────────          ──────────────────────────────
    Local training                             Waiting for 3 clients
      |  (if --compress)                             |
      v                                             v
    8-bit quantization -> float16             Receive weights (float16 or float32)
      |                                             |  maybe_decompress()
      v                                             v
    Send via Flower (gRPC)          ——>       Decompress -> float32
                                                    |
                                              L2 detection
                                                    |
                                              Aggregation (FedAvg / Krum / TrimmedMean)
                                                    |
                                              Blockchain (SHA256 hash-chain)
                                                    |
                                              Send global weights -> clients

### Run Experiment 5

Open 4 terminals in Exp_5/.

  Terminal 1 — Server:
    python server_exp5.py --run_name baseline --aggregator fedavg

  Terminals 2, 3, 4 — Clients:
    python client_exp5.py --client_id 0 --cpu_core 1 --server_ip 127.0.0.1
    python client_exp5.py --client_id 1 --cpu_core 2 --server_ip 127.0.0.1
    python client_exp5.py --client_id 2 --cpu_core 3 --server_ip 127.0.0.1

  With compression enabled:
    python client_exp5.py --client_id 0 --cpu_core 1 --server_ip 127.0.0.1 --compress
    python client_exp5.py --client_id 1 --cpu_core 2 --server_ip 127.0.0.1 --compress
    python client_exp5.py --client_id 2 --cpu_core 3 --server_ip 127.0.0.1 --compress

  With label-flipping attack on client 0:
    python client_exp5.py --client_id 0 --cpu_core 1 --server_ip 127.0.0.1 --attack_type label_flip --compress
    python client_exp5.py --client_id 1 --cpu_core 2 --server_ip 127.0.0.1 --compress
    python client_exp5.py --client_id 2 --cpu_core 3 --server_ip 127.0.0.1 --compress

### client_exp5.py arguments

    Argument        Values                   Default    Description
    ──────────────────────────────────────────────────────────────────────────
    --server_ip     IP address               required   Server IP address
    --client_id     0, 1, 2                  required   Client identifier
    --cpu_core      1, 2, 3                  1          CPU core (psutil pinning)
    --attack_type   none, label_flip,        none       Attack type
                    backdoor
    --compress      flag                     off        Enable 8-bit compression
    --bits          4, 8                     8          Quantization precision

### server_exp5.py arguments

    Argument        Values                           Default    Description
    ──────────────────────────────────────────────────────────────────────────
    --aggregator    fedavg, krum, trimmed_mean        fedavg     Aggregation algorithm
    --run_name      baseline, labelflip,              baseline   Attack configuration
                    backdoor, both

### Configurations tested for the paper

    # 1. Baseline without compression
    python server_exp5.py --run_name baseline --aggregator fedavg

    # 2. Krum + compression + label-flipping attack
    python server_exp5.py --run_name labelflip --aggregator krum

    # 3. Trimmed Mean + compression + backdoor
    python server_exp5.py --run_name backdoor --aggregator trimmed_mean

    # 4. Both attacks simultaneously
    python server_exp5.py --run_name both --aggregator krum

### Generate figures

    python plot_exp5.py

### Unit tests

    python test_compression.py

---

## Compatibility with Colleague's Original Clients

client_0_attack_pinned.py, client_1_attack_pinned.py, client_2_attack_pinned.py
are fully compatible with server_exp5.py. They send float32 weights (no compression),
which the server automatically detects and processes without decompression.

    python server_exp5.py --run_name labelflip --aggregator krum
    python client_0_attack_pinned.py --server_ip 127.0.0.1 --attack_type label_flip
    python client_1_attack_pinned.py --server_ip 127.0.0.1
    python client_2_attack_pinned.py --server_ip 127.0.0.1

---

## Deployment on Raspberry Pi 3

### CPU Pinning

    Process       Core
    ──────────────────
    Server        0
    Client 0      1
    Client 1      2
    Client 2      3

Raspberry Pi 3 has 4 cores (ARM Cortex-A53) — one per process, no contention.

### IP Address

Replace 127.0.0.1 with the local IP of the Raspberry Pi running the server:

    python client_exp5.py --client_id 0 --cpu_core 1 --server_ip 192.168.1.X --compress

### Check Flower version

    python -c "import flwr; print(flwr.__version__)"

Version must be identical on the server and all clients.

---

## Results

    Path                                       Content
    ────────────────────────────────────────────────────────────────────────
    Exp_4/results_exp4/*.json                  Per-round metrics (accuracy, time)
    Exp_4/results_exp4/figures/                Graphs generated by plot_exp4.py
    Exp_5/results_exp5/exp5_metrics_*.json     Exp 5 metrics (+ compression stats)
    Exp_5/results_exp5/exp5_blockchain_*.json  Blockchain log per configuration
    Exp_5/results_exp5/figures/                Graphs generated by plot_exp5.py

---

## Notes for the Paper

  - Round 1 excluded from all analyses (consistency with colleague's Exp 1–3)
  - TRIM=0 for Trimmed Mean with 3 clients (requires n - 2*trim >= 2)
    -> use trim=1 with >= 4 clients
  - F=1 for Krum (assumes at most 1 byzantine client out of 3)
  - Compression: 2x real ratio over the network (float32 -> float16),
    normalized RMSE < 1% in 8-bit
  - TDP: 15W used for energy estimation (Raspberry Pi 3: ~5W max in practice
    — conservative value)
