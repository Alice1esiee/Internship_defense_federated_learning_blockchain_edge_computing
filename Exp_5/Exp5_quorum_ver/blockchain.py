import hashlib
import json
import time

GENESIS_TIMESTAMP_FIXE = 0.0


class Block:
    def __init__(self, round_number, contributions, previous_hash, timestamp=None):
        self.round_number  = round_number
        self.contributions = contributions
        self.previous_hash = previous_hash
        self.timestamp     = timestamp if timestamp is not None else time.time()
        self.hash          = self._calculate_hash()

    def _calculate_hash(self):
        block_content = json.dumps({
            "round_number"  : self.round_number,
            "contributions" : self.contributions,
            "previous_hash" : self.previous_hash,
            "timestamp"     : self.timestamp
        }, sort_keys=True)
        return hashlib.sha256(block_content.encode()).hexdigest()

    def to_dict(self):
        return {
            "round_number"  : self.round_number,
            "contributions" : self.contributions,
            "previous_hash" : self.previous_hash,
            "timestamp"     : self.timestamp,
            "hash"          : self.hash
        }

    @classmethod
    def depuis_dict(cls, donnees):
        return cls(
            round_number  = donnees["round_number"],
            contributions = donnees["contributions"],
            previous_hash = donnees["previous_hash"],
            timestamp     = donnees["timestamp"],
        )


class Blockchain:
    def __init__(self, genesis_timestamp=None):
        self.chain   = []
        self.metrics = []
        genesis_block = Block(
            round_number  = 0,
            contributions = [],
            previous_hash = "0",
            timestamp     = genesis_timestamp,
        )
        self.chain.append(genesis_block)
        print("[Blockchain] Initialisee avec le bloc de genese.")

    def _hacher_contributions(self, client_updates):
        contributions = []
        for client_id, (weights, num_samples) in client_updates.items():
            weights_bytes = b"".join([w.tobytes() for w in weights])
            weights_hash  = hashlib.sha256(weights_bytes).hexdigest()
            contributions.append({
                "client_id"    : client_id,
                "weights_hash" : weights_hash,
                "num_samples"  : num_samples,
            })
        return contributions

    def add_block(self, round_number, client_updates, detection_info=None):
        t_start = time.perf_counter()
        contributions = []
        for client_id, (weights, num_samples) in client_updates.items():
            weights_bytes = b"".join([w.tobytes() for w in weights])
            weights_hash  = hashlib.sha256(weights_bytes).hexdigest()
            contribution  = {
                "client_id"    : client_id,
                "weights_hash" : weights_hash,
                "num_samples"  : num_samples,
            }
            if detection_info is not None and client_id in detection_info:
                contribution["l2_deviation"] = detection_info[client_id]["l2_deviation"]
                contribution["is_suspect"]   = detection_info[client_id]["is_suspect"]

            suspect_tag = ""
            if detection_info is not None and client_id in detection_info:
                suspect_tag = "  <-- SUSPECT" if detection_info[client_id]["is_suspect"] else ""
            print(f"[Blockchain] Client {client_id} | hash: {weights_hash[:16]}... | samples: {num_samples}{suspect_tag}")
            contributions.append(contribution)

        previous_hash = self.chain[-1].hash
        new_block     = Block(round_number, contributions, previous_hash)
        t_end         = time.perf_counter()
        overhead      = t_end - t_start
        self.chain.append(new_block)

        metrics = {
            "round"            : round_number,
            "overhead_seconds" : round(overhead, 4),
            "num_clients"      : len(contributions),
        }
        self.metrics.append(metrics)
        print(f"[Blockchain] Bloc {round_number} ajoute | hash: {new_block.hash[:16]}... | surcout: {overhead*1000:.2f}ms")
        return metrics

    def construire_bloc_candidat(self, round_number, client_updates):
        contributions = self._hacher_contributions(client_updates)
        previous_hash = self.chain[-1].hash
        bloc_candidat = Block(
            round_number  = round_number,
            contributions = contributions,
            previous_hash = previous_hash,
            timestamp     = time.time(),
        )
        print(f"[Blockchain] Bloc candidat round {round_number} propose | hash: {bloc_candidat.hash[:16]}...")
        return bloc_candidat

    def commiter_bloc(self, bloc):
        self.chain.append(bloc)
        print(f"[Blockchain] Bloc round {bloc.round_number} COMMITE | hash: {bloc.hash[:16]}...")

    def verifier_bloc_recu(self, bloc_dict, mon_client_id, mon_hash_contribution):
        bloc = Block.depuis_dict(bloc_dict)
        if bloc.hash != bloc_dict["hash"]:
            print(f"[Quorum][Client {mon_client_id}] REJET : hash du bloc incoherent.")
            return False, None
        if bloc.previous_hash != self.chain[-1].hash:
            print(f"[Quorum][Client {mon_client_id}] REJET : previous_hash ne colle pas a ma chaine locale.")
            return False, None
        ma_contribution_presente = any(
            (c["client_id"] == mon_client_id and c["weights_hash"] == mon_hash_contribution)
            for c in bloc.contributions
        )
        if not ma_contribution_presente:
            print(f"[Quorum][Client {mon_client_id}] REJET : ma contribution est absente ou modifiee.")
            return False, None
        print(f"[Quorum][Client {mon_client_id}] Bloc round {bloc.round_number} VALIDE → signature.")
        return True, bloc

    def is_valid(self):
        for i in range(1, len(self.chain)):
            current  = self.chain[i]
            previous = self.chain[i - 1]
            if current.hash != current._calculate_hash():
                print(f"[Blockchain] ERREUR : Bloc {i} modifie : hash invalide !")
                return False
            if current.previous_hash != previous.hash:
                print(f"[Blockchain] ERREUR : Lien casse entre bloc {i-1} et bloc {i} !")
                return False
        print("[Blockchain] Chaine integre : aucune falsification detectee.")
        return True

    def save(self, filepath="blockchain_ledger.json"):
        ledger = {
            "blocks"  : [block.to_dict() for block in self.chain],
            "metrics" : self.metrics
        }
        with open(filepath, "w") as f:
            json.dump(ledger, f, indent=2)
        print(f"[Blockchain] Registre sauvegarde dans {filepath}")

    def print_summary(self):
        print("\n" + "="*60)
        print("REGISTRE BLOCKCHAIN : RESUME")
        print("="*60)
        for block in self.chain[1:]:
            print(f"\nRound {block.round_number} | {time.strftime('%H:%M:%S', time.localtime(block.timestamp))}")
            print(f"  Hash bloc     : {block.hash[:32]}...")
            print(f"  Hash precedent: {block.previous_hash[:32]}...")
            for c in block.contributions:
                suspect = "  <-- SUSPECT" if c.get("is_suspect") else ""
                print(f"  Client {c['client_id']} → hash: {c['weights_hash'][:24]}... | {c['num_samples']} samples{suspect}")
        print("="*60 + "\n")


def hash_weights(weights):
    weights_bytes = b"".join([w.tobytes() for w in weights])
    return hashlib.sha256(weights_bytes).hexdigest()