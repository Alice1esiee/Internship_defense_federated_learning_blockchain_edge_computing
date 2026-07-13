"""
plot_exp4.py
Génère les figures de l'Expérience 4 à partir des résultats JSON.

Figures produites (dans results_exp4/figures/) :
  1. accuracy_vs_rounds_{alpha}_byz{n}.png  — courbes par agrégateur, par config
  2. final_accuracy_heatmap.png             — heatmap finale agrégateur × alpha × byzantins
  3. accuracy_vs_alpha.png                  — accuracy finale vs alpha par agrégateur/attaque
"""

import json
import os
from pathlib import Path
from itertools import product

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_DIR = Path("results_exp4")
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [0.1, 0.5, 1.0]
BYZ_FRACTIONS = [0, 1]
AGGREGATORS = ["fedavg", "krum", "trimmed_mean"]
AGG_LABELS = {"fedavg": "FedAvg", "krum": "Krum", "trimmed_mean": "Trimmed Mean"}
AGG_COLORS = {"fedavg": "#1f77b4", "krum": "#ff7f0e", "trimmed_mean": "#2ca02c"}
AGG_STYLES = {"fedavg": "-", "krum": "--", "trimmed_mean": "-."}

ROUND_SKIP = 1   # on exclut le round 1 comme la collègue


def load_result(alpha, num_byz, agg):
    tag = f"alpha{alpha}_byz{num_byz}_{agg}"
    path = RESULTS_DIR / f"exp4_{tag}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_accuracies(result):
    """Retourne les accuracy des rounds >= 2 (round 1 exclu)."""
    rounds = [r for r in result["rounds"] if r["round"] > ROUND_SKIP]
    return [r["round"] for r in rounds], [r["accuracy"] for r in rounds]


# ── Figure 1 : accuracy vs rounds ────────────────────────────────────────────
def plot_accuracy_vs_rounds():
    for alpha, num_byz in product(ALPHAS, BYZ_FRACTIONS):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        has_data = False

        for agg in AGGREGATORS:
            result = load_result(alpha, num_byz, agg)
            if result is None:
                continue
            rounds, accs = get_accuracies(result)
            ax.plot(rounds, accs,
                    label=AGG_LABELS[agg],
                    color=AGG_COLORS[agg],
                    linestyle=AGG_STYLES[agg],
                    linewidth=2, marker="o", markersize=5)
            has_data = True

        if not has_data:
            plt.close(fig)
            continue

        attack_label = f"{num_byz} Byzantine Client(s)" if num_byz > 0 else "No Attack"
        ax.set_title(f"CIFAR-10 Non-IID | α={alpha} | {attack_label}", fontsize=12)
        ax.set_xlabel("FL Round", fontsize=11)
        ax.set_ylabel("Global Accuracy (%)", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_ylim(0, 100)

        fname = FIG_DIR / f"accuracy_vs_rounds_alpha{alpha}_byz{num_byz}.png"
        fig.tight_layout()
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  ✓ {fname}")


# ── Figure 2 : heatmap accuracy finale ───────────────────────────────────────
def plot_heatmap():
    """
    Heatmap : ligne = (alpha, num_byz), colonne = agrégateur
    Valeur = accuracy finale (dernier round).
    """
    configs = list(product(ALPHAS, BYZ_FRACTIONS))
    n_rows = len(configs)
    n_cols = len(AGGREGATORS)

    matrix = np.full((n_rows, n_cols), np.nan)
    row_labels = [f"α={a}, byz={b}" for a, b in configs]
    col_labels = [AGG_LABELS[a] for a in AGGREGATORS]

    for i, (alpha, num_byz) in enumerate(configs):
        for j, agg in enumerate(AGGREGATORS):
            result = load_result(alpha, num_byz, agg)
            if result is None:
                continue
            _, accs = get_accuracies(result)
            if accs:
                matrix[i, j] = accs[-1]

    if np.all(np.isnan(matrix)):
        print("  ⚠ Pas de données pour la heatmap.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    plt.colorbar(im, ax=ax, label="Final Accuracy (%)")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title("Final Accuracy — CIFAR-10 Non-IID\n(Last Round, Round 1 Excluded)", fontsize=11)

    for i in range(n_rows):
        for j in range(n_cols):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.1f}",
                        ha="center", va="center", fontsize=9,
                        color="black" if 30 < matrix[i, j] < 80 else "white")

    fname = FIG_DIR / "final_accuracy_heatmap.png"
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  ✓ {fname}")


# ── Figure 3 : accuracy finale vs alpha ──────────────────────────────────────
def plot_accuracy_vs_alpha():
    """
    Pour chaque combinaison (agrégateur, num_byz) : courbe accuracy finale vs alpha.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, num_byz, title in zip(axes,
                                   BYZ_FRACTIONS,
                                   ["No Attack (byz=0)", "1 Byzantine Client (byz=1)"]):
        for agg in AGGREGATORS:
            final_accs = []
            for alpha in ALPHAS:
                result = load_result(alpha, num_byz, agg)
                if result is None:
                    final_accs.append(np.nan)
                    continue
                _, accs = get_accuracies(result)
                final_accs.append(accs[-1] if accs else np.nan)

            ax.plot(ALPHAS, final_accs,
                    label=AGG_LABELS[agg],
                    color=AGG_COLORS[agg],
                    linestyle=AGG_STYLES[agg],
                    linewidth=2, marker="s", markersize=7)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("α (Dirichlet — Heterogeneity)", fontsize=10)
        ax.set_ylabel("Final Accuracy (%)", fontsize=10)
        ax.set_xticks(ALPHAS)
        ax.set_xticklabels([str(a) for a in ALPHAS])
        ax.legend(fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_ylim(0, 100)

    fig.suptitle("Impact of Non-IID Degree (α) on Aggregator Robustness\nCIFAR-10 — Last FL Round",
                 fontsize=12)
    fname = FIG_DIR / "accuracy_vs_alpha.png"
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  ✓ {fname}")


# ── Point d'entrée ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Génération des figures Exp 4…\n")

    print("── Figure 1 : accuracy vs rounds ──")
    plot_accuracy_vs_rounds()

    print("\n── Figure 2 : heatmap accuracy finale ──")
    plot_heatmap()

    print("\n── Figure 3 : accuracy finale vs alpha ──")
    plot_accuracy_vs_alpha()

    print(f"\n✓ Toutes les figures dans : {FIG_DIR}/")