import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
import torch
import torch.nn as nn
import numpy as np
import json
import time
import os
import psutil
import argparse
from typing import List, Tuple, Optional, Dict, Union
from flwr.common import Metrics, FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy

from blockchain import Blockchain, hash_weights, GENESIS_TIMESTAMP_FIXE   # ← modifié
from aggregators import krum, trimmed_mean
from compression import CompressedWeights, dequantize_weights


NUM_ROUNDS  = 5
NUM_CLIENTS = 3
F_BYZANTINE = 1
TRIM        = 0


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


initial_model      = MNISTModel()
initial_parameters = ndarrays_to_parameters(
    [val.cpu().numpy() for val in initial_model.state_dict().values()]
)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [n * m["accuracy"] for n, m in metrics]
    examples   = [n for n, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def maybe_decompress(weights: list[np.ndarray], fit_metrics: dict) -> list[np.ndarray]:
    if weights[0].dtype != np.float16:
        return [w.astype(np.float32) for w in weights]
    mins_json = fit_metrics.get("compress_mins", None)
    maxs_json = fit_metrics.get("compress_maxs", None)
    bits      = int(fit_metrics.get("bits", 8))
    if mins_json is None or maxs_json is None:
        print("[Serveur] ⚠ Poids float16 reçus mais pas de mins/maxs — fallback float32")
        return [w.astype(np.float32) for w in weights]
    mins = json.loads(mins_json)
    maxs = json.loads(maxs_json)
    compressed = CompressedWeights(
        quantized       = [w.astype(np.int16) for w in weights],
        mins            = mins,
        maxs            = maxs,
        bits            = bits,
        original_shapes = [w.shape for w in weights],
        original_dtype  = "float32",
    )
    return dequantize_weights(compressed)


class FedAvgWithDefense(FedAvg):

    def __init__(self, blockchain, results_list, aggregator="fedavg", **kwargs):
        super().__init__(**kwargs)
        self.blockchain       = blockchain
        self.results_list     = results_list
        self.aggregator       = aggregator
        self.round_start_data = {}
        valid = {"fedavg", "krum", "trimmed_mean"}
        if aggregator not in valid:
            raise ValueError(f"--aggregator doit être dans {valid}")
        print(f"[Serveur] Agrégateur : {self.aggregator.upper()}")

    def configure_fit(self, server_round, parameters, client_manager):
        t_start   = time.time()
        cpu_avant = psutil.cpu_percent(interval=0.1)
        ram_avant = psutil.virtual_memory().used / (1024 ** 2)
        self.round_start_data[server_round] = (t_start, ram_avant, cpu_avant)
        print(f"\n[Serveur] Round {server_round} DEBUT | RAM: {ram_avant:.1f} MB | CPU: {cpu_avant:.1f}%")
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):

        t_start, ram_avant, cpu_avant = self.round_start_data.get(
            server_round, (time.time(), 0, 0)
        )

        client_ids           = []
        weights_list         = []
        samples_list         = []
        client_attack_types  = {}
        total_bytes_received = 0
        compression_stats    = {}

        for client_proxy, fit_res in results:
            raw_weights  = parameters_to_ndarrays(fit_res.parameters)
            num_samples  = fit_res.num_examples
            client_id    = fit_res.metrics.get("client_id", client_proxy.cid)
            attack_type  = fit_res.metrics.get("attack_type", "unknown")
            compress_on  = bool(fit_res.metrics.get("compress", 0))
            bits         = int(fit_res.metrics.get("bits", 8))

            bytes_received = sum(w.nbytes for w in raw_weights)
            total_bytes_received += bytes_received

            weights = maybe_decompress(raw_weights, fit_res.metrics)
            weights = [np.nan_to_num(w) for w in weights]

            client_ids.append(client_id)
            weights_list.append(weights)
            samples_list.append(num_samples)
            client_attack_types[client_id] = attack_type

            bytes_original = sum(w.nbytes for w in weights)
            compression_stats[client_id] = {
                "compress"       : compress_on,
                "bits"           : bits,
                "bytes_received" : bytes_received,
                "bytes_original" : bytes_original,
                "compress_ratio" : round(bytes_original / bytes_received, 3) if compress_on else 1.0,
            }
            print(f"[Serveur] Client {client_id} (attack={attack_type}, compress={compress_on}) "
                  f"→ {bytes_received/1024:.2f} KB reçus"
                  + (f" [ratio {compression_stats[client_id]['compress_ratio']:.2f}x]" if compress_on else ""))

        krum_selected_id = None

        if self.aggregator == "fedavg":
            # FedAvg manuel sur les poids DÉJÀ décompressés (weights_list)
            total_samples = sum(samples_list)
            aggregated_weights = [
                sum(w[i] * s / total_samples for w, s in zip(weights_list, samples_list))
                for i in range(len(weights_list[0]))
            ]
            aggregated_parameters = ndarrays_to_parameters(aggregated_weights)
            aggregated_metrics    = {}

        elif self.aggregator == "krum":
            selected_weights, best_idx = krum(weights_list, f=F_BYZANTINE)
            krum_selected_id      = client_ids[best_idx]
            aggregated_weights    = selected_weights
            aggregated_parameters = ndarrays_to_parameters(aggregated_weights)
            aggregated_metrics    = {}
            print(f"[Serveur] Krum → client sélectionné : {krum_selected_id} "
                  f"(attack={client_attack_types[krum_selected_id]})")

        elif self.aggregator == "trimmed_mean":
            aggregated_weights    = trimmed_mean(weights_list, trim=TRIM)
            aggregated_parameters = ndarrays_to_parameters(aggregated_weights)
            aggregated_metrics    = {}
            print(f"[Serveur] Trimmed Mean → trim={TRIM} "
                  f"({len(weights_list)} clients, {len(weights_list)-2*TRIM} gardés)")

        mean_weights = [
            np.mean([w[i] for w in weights_list], axis=0)
            for i in range(len(weights_list[0]))
        ]
        l2_deviations = {
            cid: sum(float(np.linalg.norm(layer - m))
                     for layer, m in zip(w, mean_weights))
            for cid, w in zip(client_ids, weights_list)
        }
        suspect_id = max(l2_deviations, key=l2_deviations.get)
        print(f"[Serveur] Déviations L2 : "
              + ", ".join(f"client {cid}={dev:.4f}" for cid, dev in l2_deviations.items()))
        print(f"[Serveur] Suspect L2 : {suspect_id} "
              f"(attack_type réel = {client_attack_types[suspect_id]})")

        client_updates = {
            cid: (w, s)
            for cid, w, s in zip(client_ids, weights_list, samples_list)
        }
        detection_info = {
            cid: {
                "l2_deviation" : round(l2_deviations[cid], 4),
                "is_suspect"   : (cid == suspect_id),
            }
            for cid in client_ids
        }
        t_bc_start            = time.perf_counter()
        blockchain_metrics    = self.blockchain.add_block(
            server_round, client_updates, detection_info
        )
        surcout_blockchain_ms = (time.perf_counter() - t_bc_start) * 1000

        t_end       = time.time()
        duree_round = t_end - t_start
        cpu_apres   = psutil.cpu_percent(interval=0.1)
        ram_apres   = psutil.virtual_memory().used / (1024 ** 2)
        delta_ram   = ram_apres - ram_avant
        cpu_moyen   = (cpu_avant + cpu_apres) / 2

        total_bytes_sent    = sum(w.nbytes for w in aggregated_weights)
        overhead_comm_bytes = total_bytes_received + (total_bytes_sent * NUM_CLIENTS)

        TDP_WATTS         = 15.0
        puissance_estimee = TDP_WATTS * (cpu_moyen / 100)
        energie_joules    = puissance_estimee * duree_round

        round_metrics = {
            "round"                    : server_round,
            "aggregator"               : self.aggregator,
            "krum_selected_id"         : krum_selected_id,
            "temps_total_secondes"     : round(duree_round, 4),
            "temps_total_ms"           : round(duree_round * 1000, 2),
            "surcout_blockchain_ms"    : round(surcout_blockchain_ms, 4),
            "cpu_avant"                : cpu_avant,
            "cpu_apres"                : cpu_apres,
            "cpu_moyen"                : round(cpu_moyen, 2),
            "ram_avant_mb"             : round(ram_avant, 2),
            "ram_apres_mb"             : round(ram_apres, 2),
            "delta_ram_mb"             : round(delta_ram, 2),
            "comm_bytes_recus"         : total_bytes_received,
            "comm_bytes_envoyes"       : total_bytes_sent * NUM_CLIENTS,
            "comm_overhead_total_bytes": overhead_comm_bytes,
            "comm_overhead_total_kb"   : round(overhead_comm_bytes / 1024, 2),
            "energie_estimee_joules"   : round(energie_joules, 4),
            "puissance_estimee_watts"  : round(puissance_estimee, 4),
            "tdp_watts_utilise"        : TDP_WATTS,
            "nb_clients"               : len(results),
            "l2_deviations"            : {cid: round(dev, 4) for cid, dev in l2_deviations.items()},
            "client_attack_types"      : client_attack_types,
            "suspect_id"               : suspect_id,
            "suspect_attack_type_reel" : client_attack_types[suspect_id],
            "detection_correcte"       : client_attack_types[suspect_id] != "none",
            "compression_stats"        : compression_stats,
            "any_client_compressed"    : any(s["compress"] for s in compression_stats.values()),
        }
        self.results_list.append(round_metrics)

        print(f"\n[Serveur] Round {server_round} FIN ({self.aggregator.upper()})")
        print(f"  Temps total         : {round_metrics['temps_total_ms']:.2f} ms")
        print(f"  Surcout blockchain  : {round_metrics['surcout_blockchain_ms']:.4f} ms")
        print(f"  Delta RAM           : {delta_ram:+.2f} MB")
        print(f"  Comm. overhead      : {round_metrics['comm_overhead_total_kb']:.2f} KB")
        print(f"  Energie estimee     : {round_metrics['energie_estimee_joules']:.4f} J")
        print(f"  Suspect L2          : client {suspect_id} "
              f"(reel: {client_attack_types[suspect_id]}) "
              f"{'✓ CORRECT' if round_metrics['detection_correcte'] else '✗ FAUX POSITIF'}")
        for cid, cs in compression_stats.items():
            if cs["compress"]:
                print(f"  Compression client {cid} : {cs['bits']}-bit | "
                      f"ratio {cs['compress_ratio']:.2f}x | "
                      f"{cs['bytes_received']/1024:.1f} KB reçus")
        if krum_selected_id:
            print(f"  Krum sélectionné    : client {krum_selected_id} "
                  f"(attack={client_attack_types[krum_selected_id]})")

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )
        accuracy = None
        if metrics_aggregated is not None:
            accuracy = metrics_aggregated.get("accuracy", None)
        for m in self.results_list:
            if m["round"] == server_round:
                m["accuracy"] = round(accuracy, 4) if accuracy is not None else None
                break
        if accuracy is not None:
            print(f"[Serveur] Accuracy globale round {server_round} : {accuracy*100:.2f}%")
        return loss_aggregated, metrics_aggregated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name",   type=str, default="baseline",
                        choices=["baseline", "labelflip", "backdoor", "both"])
    parser.add_argument("--aggregator", type=str, default="fedavg",
                        choices=["fedavg", "krum", "trimmed_mean"])
    args = parser.parse_args()

    proc_serveur = psutil.Process(os.getpid())
    proc_serveur.cpu_affinity([0])
    print(f"[Serveur] Épinglé sur cœur CPU 0")
    print(f"[Serveur] Config : run={args.run_name} | aggregator={args.aggregator}")

    os.makedirs("results_exp5", exist_ok=True)

    blockchain   = Blockchain(genesis_timestamp=GENESIS_TIMESTAMP_FIXE)   # ← modifié
    results_list = []

    strategy = FedAvgWithDefense(
        blockchain                      = blockchain,
        results_list                    = results_list,
        aggregator                      = args.aggregator,
        fraction_fit                    = 1.0,
        min_fit_clients                 = NUM_CLIENTS,
        min_available_clients           = NUM_CLIENTS,
        initial_parameters              = initial_parameters,
        evaluate_metrics_aggregation_fn = weighted_average,
    )

    print(f"=== Expérience 5 — Défenses + Compression (Gap B) ===")
    print(f"Agrégateur : {args.aggregator.upper()}")
    print(f"En attente de {NUM_CLIENTS} clients sur le port 8080...\n")

    fl.server.start_server(
        server_address = "0.0.0.0:8080",
        config         = fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy       = strategy,
    )

    blockchain.print_summary()
    blockchain.is_valid()

    metrics_path = f"results_exp5/exp5_metrics_{args.aggregator}_{args.run_name}.json"
    ledger_path  = f"results_exp5/exp5_blockchain_{args.aggregator}_{args.run_name}.json"

    blockchain.save(ledger_path)
    with open(metrics_path, "w") as f:
        json.dump(results_list, f, indent=2)

    print(f"\n[Serveur] Métriques → {metrics_path}")
    print(f"[Serveur] Registre  → {ledger_path}")

    nb_corrects = sum(1 for m in results_list if m["detection_correcte"])
    print(f"\n[Serveur] Détection L2 correcte : {nb_corrects}/{len(results_list)} rounds")

    if args.aggregator == "krum":
        krum_attacks = [
            m["client_attack_types"].get(m["krum_selected_id"], "?")
            for m in results_list if m["krum_selected_id"] is not None
        ]
        nb_krum_clean = sum(1 for a in krum_attacks if a == "none")
        print(f"[Serveur] Krum client sain sélectionné : {nb_krum_clean}/{len(results_list)} rounds")

    all_compressed = [
        cs
        for m in results_list
        for cs in m["compression_stats"].values()
        if cs["compress"]
    ]
    if all_compressed:
        avg_ratio  = np.mean([cs["compress_ratio"] for cs in all_compressed])
        total_saved = sum(cs["bytes_original"] - cs["bytes_received"] for cs in all_compressed)
        print(f"\n[Serveur] Compression — ratio moyen : {avg_ratio:.2f}x | "
              f"bandes économisées : {total_saved/1024:.1f} KB au total")