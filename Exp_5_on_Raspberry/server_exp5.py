"""
server_exp5.py
Simulation FL en mémoire — Expérience 5.
Remplace l'architecture Flower par une boucle locale.
Conserve exactement : blockchain, compression, défenses byzantines,
métriques CPU/RAM, détection L2, label-flipping, backdoor.

Lancer :
  python server_exp5.py --run_name baseline --aggregator krum
  python server_exp5.py --run_name labelflip --aggregator krum
  python server_exp5.py --run_name backdoor  --aggregator trimmed_mean
"""

import argparse
import copy
import json
import os
import time
import numpy as np
import psutil

from blockchain   import Blockchain
from aggregators  import krum, trimmed_mean
from compression  import (compress_weights, decompress_weights,
                          compression_ratio, weights_size_bytes,
                          CompressedWeights)
from exp5_data    import load_mnist, get_client_data
from exp5_model   import MNISTModel

# ── Paramètres (identiques à l'original) ─────────────────────────────────────
NUM_ROUNDS  = 5
NUM_CLIENTS = 3
F_BYZANTINE = 1
TRIM        = 0
BATCH_SIZE  = 32
LR          = 0.01
MOMENTUM    = 0.9
TDP_WATTS   = 15.0   # TDP Raspberry Pi 3 (estimation)
SEED        = 42

# ── Attaques ──────────────────────────────────────────────────────────────────

def apply_trigger(images, trigger_size=3, trigger_value=1.0):
    """Ajoute un carré blanc dans le coin bas-droit (backdoor trigger)."""
    imgs = images.copy()
    imgs[:, :, -trigger_size:, -trigger_size:] = trigger_value
    return imgs


def train_local(model, images, labels, attack_type="none", epochs=1):
    """
    Entraînement local d'une copie du modèle.
    attack_type : "none" / "label_flip" / "backdoor"
    Retourne les poids mis à jour.
    """
    m = copy.deepcopy(model)
    n = len(images)

    for _ in range(epochs):
        perm = np.random.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            idx   = perm[start:start+BATCH_SIZE]
            x_b   = images[idx]
            y_b   = labels[idx].copy()

            if attack_type == "label_flip":
                # 7 → 1 (identique à l'original)
                y_b = np.where(y_b == 7, 1, y_b)

            elif attack_type == "backdoor":
                # 50% des images reçoivent le trigger, forcé à label 0
                n_poison = max(1, len(idx) // 2)
                x_b = x_b.copy()
                x_b[:n_poison] = apply_trigger(x_b[:n_poison])
                y_b[:n_poison] = 0

            probs = m.forward(x_b, store=True)
            grads = m.backward(probs, y_b)
            m.sgd_step(grads, lr=LR, momentum=MOMENTUM)

    return m.get_weights()


def evaluate(model, test_images, test_labels):
    """Accuracy globale (%)."""
    correct = 0
    n = len(test_images)
    for start in range(0, n, 256):
        x = test_images[start:start+256]
        y = test_labels[start:start+256]
        probs = model.forward(x, store=False)
        correct += (probs.argmax(axis=1) == y).sum()
    return correct / n


def test_backdoor_asr(model, test_images, trigger_size=3, target_label=0):
    """Attack Success Rate (ASR) backdoor."""
    correct = 0
    n = len(test_images)
    for start in range(0, n, 256):
        x = apply_trigger(test_images[start:start+256], trigger_size)
        probs = model.forward(x, store=False)
        correct += (probs.argmax(axis=1) == target_label).sum()
    return correct / n


# ── Agrégation ────────────────────────────────────────────────────────────────

def fedavg(weights_list):
    return [np.mean(np.stack(layers), axis=0) for layers in zip(*weights_list)]


def aggregate(weights_list, aggregator):
    if aggregator == "fedavg":
        return fedavg(weights_list), None
    elif aggregator == "krum":
        w, idx = krum(weights_list, f=F_BYZANTINE)
        return w, idx
    elif aggregator == "trimmed_mean":
        return trimmed_mean(weights_list, trim=TRIM), None
    else:
        raise ValueError(f"Agrégateur inconnu : {aggregator}")


# ── Simulation complète ───────────────────────────────────────────────────────

def run_exp5(run_name, aggregator_name):
    """
    Simule l'Exp 5 en mémoire.
    run_name : "baseline" | "labelflip" | "backdoor"
    """
    np.random.seed(SEED)
    os.makedirs("results_exp5", exist_ok=True)

    # Attribution des attaques par client (identique à l'original)
    # client 0 = attaquant si attaque active, clients 1 et 2 = honnêtes
    attack_map = {
        "baseline"  : {0: "none",       1: "none",  2: "none"},
        "labelflip" : {0: "label_flip",  1: "none",  2: "none"},
        "backdoor"  : {0: "backdoor",    1: "none",  2: "none"},
    }
    attacks = attack_map[run_name]

    print(f"\n{'='*60}")
    print(f"  Exp 5 — run={run_name} | agrégateur={aggregator_name.upper()}")
    print(f"  Attaques : {attacks}")
    print(f"{'='*60}\n")

    # Données
    train_images, train_labels, test_images, test_labels = load_mnist()
    client_data = [
        get_client_data(train_images, train_labels, k, NUM_CLIENTS)
        for k in range(NUM_CLIENTS)
    ]

    # Modèle global + blockchain
    global_model = MNISTModel(seed=SEED)
    blockchain   = Blockchain()
    results_list = []

    for rnd in range(1, NUM_ROUNDS + 1):
        t_start  = time.time()
        cpu_avant = psutil.cpu_percent(interval=0.1)
        ram_avant = psutil.virtual_memory().used / (1024 ** 2)
        print(f"\n[Serveur] Round {rnd} DEBUT | RAM: {ram_avant:.1f} MB | CPU: {cpu_avant:.1f}%")

        # ── Entraînement local + compression ────────────────────
        client_ids      = list(range(NUM_CLIENTS))
        weights_list    = []
        samples_list    = []
        compression_stats = {}
        total_bytes_received = 0

        for k in client_ids:
            imgs, lbls = client_data[k]
            attack = attacks[k]

            # Entraînement local
            w = train_local(global_model, imgs, lbls, attack_type=attack)
            original_bytes = weights_size_bytes(w)

            # Compression 8-bit (tous les clients compressent)
            compressed     = compress_weights(w, bits=8)
            w_transport    = [q.astype(np.float16) for q in compressed.quantized]
            comp_bytes     = sum(a.nbytes for a in w_transport)
            w_decomp       = decompress_weights(compressed)

            weights_list.append(w_decomp)
            samples_list.append(len(imgs))
            total_bytes_received += comp_bytes

            compression_stats[k] = {
                "compress"       : True,
                "bits"           : 8,
                "bytes_received" : comp_bytes,
                "bytes_original" : original_bytes,
                "compress_ratio" : round(original_bytes / comp_bytes, 3),
            }
            print(f"[Client {k}] attack={attack} | "
                  f"{original_bytes/1024:.1f} KB → {comp_bytes/1024:.1f} KB "
                  f"(ratio {compression_stats[k]['compress_ratio']:.2f}x)")

        # ── Agrégation ───────────────────────────────────────────
        agg_weights, krum_idx = aggregate(weights_list, aggregator_name)
        krum_selected_id = client_ids[krum_idx] if krum_idx is not None else None
        global_model.set_weights(agg_weights)

        # ── Détection L2 ─────────────────────────────────────────
        mean_w = [np.mean(np.stack([w[i] for w in weights_list]), axis=0)
                  for i in range(len(weights_list[0]))]
        l2_devs = {
            k: sum(float(np.linalg.norm(layer - m))
                   for layer, m in zip(weights_list[k], mean_w))
            for k in range(NUM_CLIENTS)
        }
        max_id  = max(l2_devs, key=l2_devs.get)
        max_val = l2_devs[max_id]
        others  = [v for k, v in l2_devs.items() if k != max_id]
        mean_others = np.mean(others)

        L2_THRESHOLD = 2.0   # suspect si déviation > 2× la moyenne des autres
        if max_val > L2_THRESHOLD * mean_others:
            suspect_id = max_id
        else:
            suspect_id = None

        print(f"[Serveur] Déviations L2 : "
            + ", ".join(f"client {k}={v:.4f}" for k, v in l2_devs.items()))
        if suspect_id is not None:
            print(f"[Serveur] Suspect L2 : client {suspect_id} "
                f"(attaque réelle = {attacks[suspect_id]})")
        else:
            print(f"[Serveur] Suspect L2 : aucun (déviations trop proches)")

        # ── Blockchain ───────────────────────────────────────────
        client_updates = {k: (weights_list[k], samples_list[k]) for k in client_ids}
        detection_info = {
            k: {"l2_deviation": round(l2_devs[k], 4),
                "is_suspect"  : (suspect_id is not None and k == suspect_id)}
            for k in client_ids
        }
        t_bc = time.perf_counter()
        blockchain.add_block(rnd, client_updates, detection_info)
        surcout_bc_ms = (time.perf_counter() - t_bc) * 1000

        # ── Évaluation globale ───────────────────────────────────
        acc = evaluate(global_model, test_images, test_labels)
        asr = None
        if run_name == "backdoor":
            asr = test_backdoor_asr(global_model, test_images)
            print(f"[Serveur] ASR backdoor : {asr*100:.2f}%")

        # ── Métriques ────────────────────────────────────────────
        t_end        = time.time()
        duree_round  = t_end - t_start
        cpu_apres    = psutil.cpu_percent(interval=0.1)
        ram_apres    = psutil.virtual_memory().used / (1024 ** 2)
        delta_ram    = ram_apres - ram_avant
        cpu_moyen    = (cpu_avant + cpu_apres) / 2

        total_bytes_sent    = weights_size_bytes(agg_weights)
        overhead_comm_bytes = total_bytes_received + (total_bytes_sent * NUM_CLIENTS)
        puissance           = TDP_WATTS * (cpu_moyen / 100)
        energie             = puissance * duree_round

        round_metrics = {
            "round"                     : rnd,
            "aggregator"                : aggregator_name,
            "krum_selected_id"          : krum_selected_id,
            "accuracy"                  : round(float(acc), 4),
            "backdoor_asr"              : round(float(asr), 4) if asr is not None else None,
            "temps_total_secondes"      : round(duree_round, 4),
            "temps_total_ms"            : round(duree_round * 1000, 2),
            "surcout_blockchain_ms"     : round(surcout_bc_ms, 4),
            "cpu_avant"                 : cpu_avant,
            "cpu_apres"                 : cpu_apres,
            "cpu_moyen"                 : round(cpu_moyen, 2),
            "ram_avant_mb"              : round(ram_avant, 2),
            "ram_apres_mb"              : round(ram_apres, 2),
            "delta_ram_mb"              : round(delta_ram, 2),
            "comm_bytes_recus"          : total_bytes_received,
            "comm_bytes_envoyes"        : total_bytes_sent * NUM_CLIENTS,
            "comm_overhead_total_bytes" : overhead_comm_bytes,
            "comm_overhead_total_kb"    : round(overhead_comm_bytes / 1024, 2),
            "energie_estimee_joules"    : round(energie, 4),
            "puissance_estimee_watts"   : round(puissance, 4),
            "tdp_watts_utilise"         : TDP_WATTS,
            "nb_clients"                : NUM_CLIENTS,
            "l2_deviations"             : {k: round(v, 4) for k, v in l2_devs.items()},
            "client_attack_types"       : attacks,
            "suspect_id"                : suspect_id,
            "suspect_attack_type_reel"  : attacks[suspect_id] if suspect_id is not None else "none",
            "detection_correcte"        : (suspect_id is not None and attacks[suspect_id] != "none"),
            "compression_stats"         : compression_stats,
            "any_client_compressed"     : True,
        }
        results_list.append(round_metrics)

        print(f"\n[Serveur] Round {rnd} FIN ({aggregator_name.upper()})")
        print(f"  Accuracy : {acc*100:.2f}%")
        print(f"  Temps    : {duree_round*1000:.1f} ms")
        print(f"  BC cost  : {surcout_bc_ms:.4f} ms")
        print(f"  RAM Δ    : {delta_ram:+.2f} MB")
        print(f"  Comm KB  : {overhead_comm_bytes/1024:.2f}")
        print(f"  Énergie  : {energie:.4f} J")
        if suspect_id is not None:
            status = "✓ CORRECT" if round_metrics["detection_correcte"] else "✗ FAUX POSITIF"
            print(f"  Suspect  : client {suspect_id} ({status})")
        else:
            print(f"  Suspect  : aucun (✓ CORRECT — pas d'attaque détectée)")

    # ── Post-traitement ───────────────────────────────────────────────────────
    blockchain.print_summary()
    blockchain.is_valid()

    metrics_path = f"results_exp5/exp5_metrics_{aggregator_name}_{run_name}.json"
    ledger_path  = f"results_exp5/exp5_blockchain_{aggregator_name}_{run_name}.json"

    blockchain.save(ledger_path)
    with open(metrics_path, "w") as f:
        json.dump(results_list, f, indent=2)

    print(f"\n[Serveur] Métriques → {metrics_path}")
    print(f"[Serveur] Registre  → {ledger_path}")

    nb_corrects = sum(1 for m in results_list if m["detection_correcte"])
    print(f"[Serveur] Détection L2 correcte : {nb_corrects}/{len(results_list)} rounds")

    avg_ratio = np.mean([
        cs["compress_ratio"]
        for m in results_list
        for cs in m["compression_stats"].values()
    ])
    print(f"[Serveur] Ratio compression moyen : {avg_ratio:.2f}x")


# ── Point d'entrée ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name",   type=str, default="baseline",
                        choices=["baseline", "labelflip", "backdoor"])
    parser.add_argument("--aggregator", type=str, default="krum",
                        choices=["fedavg", "krum", "trimmed_mean"])
    args = parser.parse_args()

    run_exp5(args.run_name, args.aggregator)