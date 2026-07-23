# Federated Learning & Blockchain for Edge Computing
## Byzantine Defenses & Compression

Implementation and experimental evaluation of a Federated Learning (FL) framework secured by Blockchain, resilient against Byzantine attacks, and optimized for Edge Computing — PC simulations and physical deployment on Raspberry Pi 3.

> Experiments 1–3 (attacks + blockchain) were carried out by a colleague. This repo covers **Experiments 4 and 5**.

---

## Repository Structure

```text
.
├── exp4/
│   ├── data/                        # Local datasets (ignored by Git)
│   ├── results_exp4/                # JSON metrics and output plots
│   ├── aggregators.py               # Defense aggregators (Krum, Trimmed Mean) + utilities
│   ├── exp4_data.py                 # Non-IID Dirichlet data partitioning
│   ├── exp4_simulation.py           # In-memory simulation (no Flower)
│   ├── plot_exp4.py                 # Figure generation script
│   ├── test_aggregators.py          # Aggregator unit tests
│   └── test_realistic.py            # Tests on real CNN weights
│
├── exp5/
│   ├── data/                        # Local datasets (ignored by Git)
│   ├── results_exp5/                # JSON metrics and output plots
│   ├── aggregators.py               # Copy from Exp 4
│   ├── blockchain.py                # Local blockchain ledger, Quorum verification, SHA256 hash-chain (colleague)
│   ├── compression.py               # Uniform quantization module (8-bit / 4-bit)
│   ├── client_exp5.py               # Unified Flower client (replaces 3 separate files)
│   ├── client_0_attack_pinned.py    # Colleague's original clients (compatible with server_exp5.py)
│   ├── client_1_attack_pinned.py
│   ├── client_2_attack_pinned.py
│   ├── server_exp5.py               # Flower server with defenses + decompression
│   ├── plot_exp5.py                 # Figure generation script
│   └── test_compression.py          # Compression unit tests
│
├── .gitignore
└── README.md
```

---

## Prerequisites & Installation
Requirements
Python 3.9+

### Quick Setup
Clone the repository and install all required packages using the requirements.txt file:

```bash
git clone https://github.com/Alice1esiee/Internship_defense_federated_learning_blockchain_edge_computing.git
cd Internship_defense_federated_learning_blockchain_edge_computing
```

### Installation

If you are : 

**On PC:**
```bash
pip install -r requirements.txt
```

Or 

**On Raspberry Pi 3 (CPU-only PyTorch):**

```bash
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
pip install -r requirements.txt
```

> **Datasets:** CIFAR-10 and MNIST are downloaded automatically to `data/` on first run.  
> **Raspberry Pi 3:** use the CPU-only build of PyTorch (no CUDA).  
> **Flower:** version must match between server and all clients — check with `pip show flwr`.

---

## Experiment 4 — Byzantine Robustness under Non-IID Data (CIFAR-10)

**Objective:** Compare FedAvg, Krum, and Trimmed Mean on CIFAR-10 with a Non-IID Dirichlet partition (α ∈ {0.1, 0.5, 1.0}) and Byzantine clients performing label-flipping. Full in-memory simulation, no Flower, independent of the colleague's pipeline.

### PC Simulation

```bash
cd exp4
python exp4_simulation.py
```

Tunable parameters in `exp4_simulation.py`:

| Parameter | Default | Description |
|---|---|---|
| `NUM_ROUNDS` | 3 | Number of FL rounds |
| `ALPHAS` | [0.1, 0.5, 1.0] | Non-IID degree (Dirichlet α) |
| `BYZANTINE_FRACTIONS` | [0, 1] | Number of Byzantine clients |
| `AGGREGATORS` | fedavg, krum, trimmed_mean | Aggregation algorithms tested |
| `NUM_CLIENTS` | 3 | Number of clients |

Results are saved to `results_exp4/` as JSON files — one per `(alpha, byzantine, aggregator)` combination, plus a global summary file.

> **Note:** Round 1 is excluded from all analyses (consistent with colleague's Exp 1–3).

### Raspberry Pi 3 Deployment

Full experiments across all α values are too slow to run in a single session on the ARM Cortex-A53. Run one α at a time using the `--alpha` argument, then collect the JSON results before plotting:

```bash
# Session 1
python exp4_simulation.py --alpha 0.1

# Session 2
python exp4_simulation.py --alpha 0.5

# Session 3
python exp4_simulation.py --alpha 1.0
```

Each session produces individual JSON files in `results_exp4/` (e.g. `exp4_alpha0.1_byz0_fedavg.json`) and a summary file `exp4_all_results_alpha0.1.json`. Once all three sessions are complete, run `plot_exp4.py` to generate the figures.

### Generate Figures

```bash
python plot_exp4.py
```

Figures saved to `results_exp4/figures/`:

| File | Description |
|---|---|
| `accuracy_vs_rounds_alpha{a}_byz{n}.png` | Accuracy curves per aggregator |
| `final_accuracy_heatmap.png` | Final accuracy heatmap |
| `accuracy_vs_alpha.png` | Impact of Non-IID degree |

### Unit Tests

```bash
python test_aggregators.py
python test_realistic.py
```

---

## Experiment 5 — Compression + Byzantine Defenses + Blockchain (MNIST)

**Objective:** Integrate 8-bit weight quantization into the colleague's Flower pipeline, alongside Krum/Trimmed Mean defenses and a SHA256 blockchain ledger.

### Pipeline Architecture

```
Client (client_exp5.py)
  └─ Local training
  └─ [if --compress] 8-bit quantization → cast to float16 (2 bytes vs 4 bytes → 2x bandwidth saving)
  └─ Send weights + compression metadata (mins/maxs) via Flower (gRPC)
        │
        ▼
Server (server_exp5.py)
  └─ Receive weights (float16 if compressed, float32 otherwise — auto-detected)
  └─ maybe_decompress() → float32
  └─ L2 deviation detection (flag suspect client)
  └─ Aggregation (FedAvg / Krum / Trimmed Mean)
  └─ Blockchain logging (SHA256 hash-chain)
  └─ Broadcast global weights → clients (always float32)
```

### Raspberry Pi 3 Hardware Setup

| Node | Role | Processes |
|---|---|---|
| Pi 1 | Server + Client 0 | `server_exp5.py` + `client_exp5.py --client_id 0` |
| Pi 2 | Client 1 | `client_exp5.py --client_id 1` |
| Pi 3 | Client 2 | `client_exp5.py --client_id 2` |

CPU pinning (4 cores, ARM Cortex-A53 — one process per core, no contention):

| Process | Core |
|---|---|
| Server | 0 |
| Client 0 | 1 |
| Client 1 | 2 |
| Client 2 | 3 |

For physical deployment, replace `127.0.0.1` with the server Pi's local IP:

```bash
python client_exp5.py --client_id 0 --cpu_core 1 --server_ip 192.168.1.X --compress
```

### Execution Scenarios

Open 4 terminals in `exp5/`.

---

**Scenario A — Baseline (all honest clients)**

```bash
# Terminal 1 — Server
python server_exp5.py --aggregator fedavg --run_name baseline

# Terminal 2 — Client 0
python client_exp5.py --client_id 0 --cpu_core 1 --server_ip 127.0.0.1

# Terminal 3 — Client 1
python client_exp5.py --client_id 1 --cpu_core 2 --server_ip 127.0.0.1

# Terminal 4 — Client 2
python client_exp5.py --client_id 2 --cpu_core 3 --server_ip 127.0.0.1
```

With compression:

```bash
python client_exp5.py --client_id 0 --cpu_core 1 --server_ip 127.0.0.1 --compress
python client_exp5.py --client_id 1 --cpu_core 2 --server_ip 127.0.0.1 --compress
python client_exp5.py --client_id 2 --cpu_core 3 --server_ip 127.0.0.1 --compress
```

---

**Scenario B — Label Flipping attack + Krum defense**

```bash
# Server
python server_exp5.py --aggregator krum --run_name labelflip

# Client 0 (attacker — label flip: 7 → 1)
python client_exp5.py --client_id 0 --cpu_core 1 --server_ip 127.0.0.1 --attack_type label_flip --compress

# Client 1 (honest)
python client_exp5.py --client_id 1 --cpu_core 2 --server_ip 127.0.0.1 --compress

# Client 2 (honest)
python client_exp5.py --client_id 2 --cpu_core 3 --server_ip 127.0.0.1 --compress
```

---

**Scenario C — Backdoor attack + Trimmed Mean defense**

```bash
# Server
python server_exp5.py --aggregator trimmed_mean --run_name backdoor

# Client 0 (attacker — backdoor trigger, target label 0)
python client_exp5.py --client_id 0 --cpu_core 1 --server_ip 127.0.0.1 --attack_type backdoor --compress

# Client 1 (honest)
python client_exp5.py --client_id 1 --cpu_core 2 --server_ip 127.0.0.1 --compress

# Client 2 (honest)
python client_exp5.py --client_id 2 --cpu_core 3 --server_ip 127.0.0.1 --compress
```

---

**Scenario D — Combined attack (Label Flip + Backdoor) + Krum defense**

```bash
# Server
python server_exp5.py --aggregator krum --run_name both

# Client 0 (label flip)
python client_exp5.py --client_id 0 --cpu_core 1 --server_ip 127.0.0.1 --attack_type label_flip --compress

# Client 1 (backdoor)
python client_exp5.py --client_id 1 --cpu_core 2 --server_ip 127.0.0.1 --attack_type backdoor --compress

# Client 2 (honest)
python client_exp5.py --client_id 2 --cpu_core 3 --server_ip 127.0.0.1 --compress
```

---

### CLI Reference

**`client_exp5.py`**

| Argument | Required | Values | Default | Description |
|---|---|---|---|---|
| `--server_ip` | ✓ | any IP | — | Server IP address |
| `--client_id` | ✓ | 0, 1, 2 | — | Client identifier |
| `--cpu_core` | | 1, 2, 3 | 1 | CPU core (psutil pinning) |
| `--attack_type` | | none, label_flip, backdoor | none | Attack type |
| `--compress` | | flag | off | Enable 8-bit quantization before sending |
| `--bits` | | 4, 8 | 8 | Quantization precision |

**`server_exp5.py`**

| Argument | Values | Default | Description |
|---|---|---|---|
| `--aggregator` | fedavg, krum, trimmed_mean | fedavg | Aggregation algorithm |
| `--run_name` | baseline, labelflip, backdoor, both | baseline | Experiment label (used in output filenames) |

### Alternative: Pinned Client Scripts

The pre-configured scripts `client_0_attack_pinned.py`, `client_1_attack_pinned.py`, `client_2_attack_pinned.py` are fully compatible with `server_exp5.py`. They send float32 weights (no compression); the server detects this automatically via dtype check and skips decompression.

```bash
# Pi 1
python client_0_attack_pinned.py --server_ip 127.0.0.1 --attack_type label_flip

# Pi 2
python client_1_attack_pinned.py --server_ip 127.0.0.1

# Pi 3
python client_2_attack_pinned.py --server_ip 127.0.0.1
```

### Generate Figures

```bash
cd exp5
python plot_exp5.py
```

Figures saved to `results_exp5/figures/` (accuracy, communication overhead, execution times across all scenarios).

### Unit Tests

```bash
python test_compression.py
```

---

## Output Files

| Path | Contents |
|---|---|
| `exp4/results_exp4/exp4_{tag}.json` | Per-round metrics per `(alpha, byzantine, aggregator)` combination |
| `exp4/results_exp4/exp4_all_results_alpha{a}.json` | All results for a given α (one file per Pi session) |
| `exp4/results_exp4/figures/` | Figures from `plot_exp4.py` |
| `exp5/results_exp5/exp5_metrics_{aggregator}_{run_name}.json` | Per-round metrics + compression stats |
| `exp5/results_exp5/exp5_blockchain_{aggregator}_{run_name}.json` | Blockchain ledger per scenario |
| `exp5/results_exp5/figures/` | Figures from `plot_exp5.py` |

---

## Notes for the Paper

- Round 1 is excluded from all analyses (consistent with colleague's Exp 1–3).
- `TRIM=0` for Trimmed Mean with 3 clients (constraint: n − 2·trim ≥ 2) → use `trim=1` with ≥ 4 clients.
- `F=1` for Krum (assumes at most 1 Byzantine client out of 3).
- Compression achieves a real 2× ratio over the network (float32 → float16); normalized RMSE < 1% in 8-bit.
- TDP set to 15 W for energy estimation (Raspberry Pi 3 peaks at ~5 W in practice — conservative value).
