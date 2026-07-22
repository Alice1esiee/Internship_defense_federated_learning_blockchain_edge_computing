"""
plot_exp5.py
Génère les figures de l'Expérience 5 à partir des résultats JSON.

Figures produites dans results_exp5/figures/ :
  1. accuracy_vs_rounds.png       — accuracy par agrégateur sous label_flip
  2. aggregator_comparison.png    — FedAvg vs Krum vs TrimmedMean (bar chart)
  3. compression_overhead.png     — comm overhead avec/sans compression
  4. blockchain_overhead.png      — surcoût blockchain par round
"""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results_exp5")
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ROUND_SKIP = 1  # on exclut le round 1 comme la collègue

AGG_LABELS = {"fedavg": "FedAvg", "krum": "Krum", "trimmed_mean": "Trimmed Mean"}
AGG_COLORS = {"fedavg": "#d62728", "krum": "#1f77b4", "trimmed_mean": "#ff7f0e"}
AGG_STYLES = {"fedavg": "-", "krum": "--", "trimmed_mean": "-."}


def load(aggregator, run_name):
    path = RESULTS_DIR / f"exp5_metrics_{aggregator}_{run_name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_rounds_data(results, key):
    """Retourne les valeurs d'une clé pour les rounds >= 2."""
    return [
        (r["round"], r.get(key))
        for r in results
        if r["round"] > ROUND_SKIP and r.get(key) is not None
    ]


# ── Figure 1 : accuracy vs rounds (label_flip) ───────────────────────────────
def plot_accuracy_vs_rounds():
    fig, ax = plt.subplots(figsize=(8, 5))
    has_data = False

    for agg in ["fedavg", "krum", "trimmed_mean"]:
        data = load(agg, "labelflip")
        if data is None:
            continue
        rounds_accs = get_rounds_data(data, "accuracy")
        if not rounds_accs:
            continue
        rounds = [r for r, _ in rounds_accs]
        accs   = [a * 100 for _, a in rounds_accs]
        ax.plot(rounds, accs,
                label=AGG_LABELS[agg],
                color=AGG_COLORS[agg],
                linestyle=AGG_STYLES[agg],
                linewidth=2, marker="o", markersize=6)
        has_data = True

    if not has_data:
        print("  ⚠ Pas de données pour Figure 1")
        plt.close(fig)
        return

    ax.set_title("Global Accuracy Under Label-Flipping Attack\nMNIST — Experiment 5 (8-bit Compression)", fontsize=12)
    ax.set_xlabel("FL Round", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    fname = FIG_DIR / "accuracy_vs_rounds.png"
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  ✓ {fname}")


# ── Figure 2 : bar chart comparaison agrégateurs ─────────────────────────────
def plot_aggregator_comparison():
    """
    Bar chart : accuracy finale (round 5) pour chaque agrégateur,
    sous baseline, label_flip et backdoor.
    """
    configs = [
        ("baseline", "No Attack"),
        ("labelflip", "Label-Flipping"),
        ("backdoor",  "Backdoor"),
        ("both",      "Both Attacks"),
    ]
    aggregators = ["fedavg", "krum", "trimmed_mean"]

    # Récupère accuracy finale pour chaque (agg, config)
    matrix = {}
    for agg in aggregators:
        matrix[agg] = {}
        for run, label in configs:
            data = load(agg, run)
            if data is None:
                matrix[agg][run] = None
                continue
            accs = [r["accuracy"] for r in data if r.get("accuracy") is not None]
            matrix[agg][run] = accs[-1] * 100 if accs else None

    # Filtre les configs qui ont au moins une donnée
    valid_configs = [(run, lbl) for run, lbl in configs
                     if any(matrix[agg].get(run) is not None for agg in aggregators)]

    if not valid_configs:
        print("  ⚠ Pas de données pour Figure 2")
        return

    x = np.arange(len(valid_configs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, agg in enumerate(aggregators):
        vals = [matrix[agg].get(run) for run, _ in valid_configs]
        vals_plot = [v if v is not None else 0 for v in vals]
        bars = ax.bar(x + i * width, vals_plot, width,
                      label=AGG_LABELS[agg],
                      color=AGG_COLORS[agg],
                      alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            if val is not None:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.8,
                        f"{val:.1f}%", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

    ax.set_title("Final Accuracy (Round 5) by Aggregator and Configuration\nMNIST — Experiment 5 (8-bit Compression)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xticks(x + width)
    ax.set_xticklabels([lbl for _, lbl in valid_configs], fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    fname = FIG_DIR / "aggregator_comparison.png"
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  ✓ {fname}")


# ── Figure 3 : overhead comm avec/sans compression ───────────────────────────
def plot_compression_overhead():
    """
    Montre le gain de bande passante apporté par la compression.
    Compare comm_overhead sans compression (théorique float32) vs avec (float16).
    """
    data = load("krum", "labelflip")
    if data is None:
        print("  ⚠ Pas de données pour Figure 3 (krum_labelflip manquant)")
        return

    rounds = [r["round"] for r in data if r["round"] > ROUND_SKIP]
    overhead_compressed   = [r["comm_overhead_total_kb"] for r in data if r["round"] > ROUND_SKIP]
    # Théorique sans compression : float32 = 2x float16
    overhead_uncompressed = [kb * 2 for kb in overhead_compressed]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, overhead_uncompressed,
            label="Without Compression (float32)", color="#d62728",
            linewidth=2, marker="s", markersize=6, linestyle="--")
    ax.plot(rounds, overhead_compressed,
            label="With 8-bit Compression (float16)", color="#1f77b4",
            linewidth=2, marker="o", markersize=6)
    ax.fill_between(rounds, overhead_compressed, overhead_uncompressed,
                    alpha=0.15, color="#2ca02c", label="Bandwidth Savings")

    ax.set_title("Communication Overhead: Impact of Compression\nMNIST — Krum + Label-Flipping", fontsize=12)
    ax.set_xlabel("FL Round", fontsize=11)
    ax.set_ylabel("Total Overhead (KB)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Annotation du gain moyen
    gain_moyen = np.mean([u - c for u, c in zip(overhead_uncompressed, overhead_compressed)])
    ax.annotate(f"Avg. savings: {gain_moyen:.0f} KB/round\n(2x ratio)",
                xy=(rounds[len(rounds)//2], overhead_compressed[len(rounds)//2]),
                xytext=(rounds[1], overhead_uncompressed[0] * 0.6),
                arrowprops=dict(arrowstyle="->", color="green"),
                fontsize=10, color="green")

    fname = FIG_DIR / "compression_overhead.png"
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  ✓ {fname}")


# ── Figure 4 : surcoût blockchain ────────────────────────────────────────────
def plot_blockchain_overhead():
    """
    Surcoût blockchain (ms) par round pour les 3 agrégateurs sous label_flip.
    Montre que le surcoût est stable et négligeable (~2ms).
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    has_data = False

    for agg in ["fedavg", "krum", "trimmed_mean"]:
        data = load(agg, "labelflip")
        if data is None:
            continue
        rounds   = [r["round"] for r in data if r["round"] > ROUND_SKIP]
        bc_costs = [r["surcout_blockchain_ms"] for r in data if r["round"] > ROUND_SKIP]
        ax.plot(rounds, bc_costs,
                label=AGG_LABELS[agg],
                color=AGG_COLORS[agg],
                linestyle=AGG_STYLES[agg],
                linewidth=2, marker="^", markersize=6)
        has_data = True

    if not has_data:
        print("  ⚠ Pas de données pour Figure 4")
        plt.close(fig)
        return

    ax.axhline(y=2.0, color="gray", linestyle=":", alpha=0.6, label="2ms Reference")
    ax.set_title("Blockchain Overhead per Round (ms)\nMNIST — Experiment 5 (Label-Flipping)", fontsize=12)
    ax.set_xlabel("FL Round", fontsize=11)
    ax.set_ylabel("Blockchain Overhead (ms)", fontsize=11)
    ax.set_ylim(0, 5)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    fname = FIG_DIR / "blockchain_overhead.png"
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  ✓ {fname}")


# ── Point d'entrée ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Génération des figures Exp 5...\n")

    print("── Figure 1 : accuracy vs rounds (label_flip) ──")
    plot_accuracy_vs_rounds()

    print("\n── Figure 2 : comparaison agrégateurs (bar chart) ──")
    plot_aggregator_comparison()

    print("\n── Figure 3 : overhead compression ──")
    plot_compression_overhead()

    print("\n── Figure 4 : surcoût blockchain ──")
    plot_blockchain_overhead()

    print(f"\n✓ Toutes les figures dans : results_exp5/figures/")