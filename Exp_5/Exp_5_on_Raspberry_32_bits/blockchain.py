# ============================================================
# blockchain.py : VERSION COMMENTÉE LIGNE PAR LIGNE
# ============================================================
# Ce fichier crée et gère la blockchain de consortium légère
# utilisée pour enregistrer les contributions de chaque client
# FL à chaque round d'entraînement.
#
# COMMENT ÇA MARCHE EN RÉSUMÉ :
# À chaque round FL, le serveur reçoit les poids de chaque client.
# blockchain.py calcule l'empreinte (hash) de ces poids et les
# enregistre dans un registre immuable : la blockchain.
# Si un client envoie des poids corrompus, on peut le prouver
# après coup grâce à ce registre.
#
# CE FICHIER EST IMPORTÉ UNIQUEMENT PAR server_blockchain.py
# Les clients (client.py) ne savent pas que ce fichier existe.
# ============================================================


# -------- IMPORTS --------

# hashlib : bibliothèque PYTHON STANDARD (préinstallée, pas de pip)
# contient les fonctions de hachage cryptographique dont SHA256
# SHA256 : prend n'importe quel texte/données → produit toujours
# 64 caractères hexadécimaux uniques (l'empreinte)
# si les données changent d'un seul bit → l'empreinte change complètement
import hashlib

# json : bibliothèque PYTHON STANDARD (préinstallée, pas de pip)
# permet de convertir des dictionnaires Python en texte JSON
# et de sauvegarder ce texte dans des fichiers .json lisibles
import json

# time : bibliothèque PYTHON STANDARD (préinstallée, pas de pip)
# time.time() → retourne l'heure actuelle en secondes (timestamp Unix)
# ex : 1715000000.123 = nombre de secondes depuis le 1er janvier 1970
# on l'utilise pour horodater chaque bloc et mesurer le surcoût
import time


# ============================================================
# CLASSE Block : définie par MOI
# représente UN round FL dans la blockchain
# ============================================================

class Block:
    # Block : classe que J'AI DÉFINIE MOI-MÊME
    # un bloc = un round FL complet
    # il contient les contributions de tous les clients pour ce round
    # plus un lien mathématique vers le bloc précédent (previous_hash)
    #
    # C'EST CE LIEN qui crée la "chaîne" :
    # si quelqu'un modifie un vieux bloc → son hash change
    # → il ne correspond plus au previous_hash du bloc suivant
    # → la chaîne "casse" → falsification détectée par is_valid()

    def __init__(self, round_number, contributions, previous_hash):
        # __init__() : constructeur que J'AI DÉFINI
        # appelé automatiquement quand on crée un bloc avec Block(...)
        # les 3 paramètres qu'on doit passer à chaque création de bloc :

        # round_number : le numéro du round FL que ce bloc représente
        # ex : 1, 2, 3, 4, 5
        self.round_number = round_number

        # contributions : liste des contributions de chaque client ce round
        # format : [{"client_id": 0, "weights_hash": "a3f9...", "num_samples": 20000}, ...]
        # weights_hash = l'empreinte SHA256 des poids envoyés par ce client
        self.contributions = contributions

        # previous_hash : l'empreinte du bloc précédent dans la chaîne
        # c'est ce lien qui rend la blockchain immuable
        # pour le tout premier bloc (bloc de genèse) → on met "0" par convention
        self.previous_hash = previous_hash

        # time.time() : fonction PYTHON STANDARD
        # enregistre l'heure exacte de création de ce bloc
        # ex : 1715000042.891 (secondes depuis 1970)
        self.timestamp = time.time()

        # _calculate_hash() : méthode que J'AI DÉFINIE plus bas
        # calcule l'empreinte de CE bloc à partir de tout son contenu
        # si n'importe quel champ change → cette empreinte change complètement
        self.hash = self._calculate_hash()

    def _calculate_hash(self):
        # _calculate_hash() : méthode que J'AI DÉFINIE MOI-MÊME
        # calcule l'empreinte SHA256 de tout le contenu de CE bloc
        # appelée dans __init__() à la création du bloc
        # et dans is_valid() pour vérifier que le bloc n'a pas été modifié
        #
        # le underscore _ devant le nom = convention Python pour dire
        # "cette méthode est interne, elle n'est pas appelée de l'extérieur"

        # json.dumps() : fonction PYTHON STANDARD
        # convertit le dictionnaire Python en chaîne de texte JSON
        # ex : {"round_number": 1, "timestamp": 1715000042.891, ...}
        # sort_keys=True : trie les clés alphabétiquement
        # POURQUOI : sans ça, {"a":1,"b":2} et {"b":2,"a":1} donneraient
        # des hashs différents alors qu'ils contiennent la même information
        block_content = json.dumps({
            "round_number"  : self.round_number,   # numéro du round
            "contributions" : self.contributions,  # contributions des clients
            "previous_hash" : self.previous_hash,  # lien vers le bloc précédent
            "timestamp"     : self.timestamp        # heure de création
        }, sort_keys=True)

        # .encode() : méthode PYTHON STANDARD sur une chaîne de texte
        # convertit le texte en octets (bytes) car SHA256 travaille sur des octets
        # ex : "hello" → b"hello"

        # hashlib.sha256() : fonction PYTHON STANDARD (bibliothèque hashlib)
        # prend les octets et calcule l'empreinte SHA256
        # .hexdigest() : retourne l'empreinte sous forme de 64 caractères hexadécimaux
        # ex : "a3f9b2c1d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0"
        return hashlib.sha256(block_content.encode()).hexdigest()

    def to_dict(self):
        # to_dict() : méthode que J'AI DÉFINIE MOI-MÊME
        # convertit le bloc en dictionnaire Python
        # utilisée dans save() pour sauvegarder la blockchain en JSON
        return {
            "round_number"  : self.round_number,
            "contributions" : self.contributions,
            "previous_hash" : self.previous_hash,
            "timestamp"     : self.timestamp,
            "hash"          : self.hash
        }


# ============================================================
# CLASSE Blockchain : définie par MOI
# la chaîne complète de tous les blocs (un par round FL)
# ============================================================

class Blockchain:
    # Blockchain : classe que J'AI DÉFINIE MOI-MÊME
    # elle contient la liste ordonnée de tous les blocs
    # et gère les métriques de performance pour l'Expérience 1

    def __init__(self):
        # __init__() : constructeur que J'AI DÉFINI
        # appelé quand on crée la blockchain avec Blockchain()
        # initialise la chaîne avec le bloc de genèse

        # self.chain : liste PYTHON qui contient tous les blocs
        # list() : PYTHON STANDARD
        # commence vide, on y ajoute un bloc après chaque round
        self.chain = []

        # self.metrics : liste PYTHON des mesures de performance
        # une entrée par round : {"round": 1, "overhead_seconds": 0.003, ...}
        self.metrics = []

        # on crée le bloc de genèse : le tout premier bloc de la chaîne
        # Block() : classe que J'AI DÉFINIE juste au-dessus
        # round_number=0 : ce n'est pas un vrai round FL, c'est juste le départ
        # contributions=[] : pas de contributions dans le bloc de genèse
        # previous_hash="0" : pas de prédécesseur → "0" par convention
        genesis_block = Block(
            round_number  = 0,
            contributions = [],
            previous_hash = "0"
        )

        # .append() : méthode PYTHON STANDARD sur une liste
        # ajoute le bloc de genèse à la chaîne
        self.chain.append(genesis_block)

        # print() : PYTHON STANDARD : affiche un message dans le terminal
        print("[Blockchain] Initialisée avec le bloc de genèse.")

    def add_block(self, round_number, client_updates, detection_info=None):
        # add_block() : méthode que J'AI DÉFINIE MOI-MÊME
        # ajoute un nouveau bloc à la chaîne après chaque round FL
        # appelée par server_blockchain_pinned.py (Expérience 1) et
        # server_attack_blockchain_pinned.py (Expérience 3) dans aggregate_fit()
        #
        # ÉTAPE PAR ÉTAPE :
        # 1. Pour chaque client → calcule SHA256 de ses poids
        # 2. Crée un nouveau bloc avec ces empreintes
        # 3. Mesure le temps que ça prend (surcoût pour Expérience 1)
        # 4. Ajoute le bloc à la chaîne
        # 5. Retourne les métriques au serveur
        #
        # round_number : numéro du round qui vient de se terminer (1, 2, 3...)
        # client_updates : dictionnaire reçu du serveur
        # format : {"client_id_0": (weights, num_samples), "client_id_1": (...), ...}
        #
        # detection_info : NOUVEAU (Expérience 3), optionnel, par défaut None
        # (donc Expérience 1 continue de fonctionner sans rien changer)
        # format : {"client_id_0": {"l2_deviation": 0.87, "is_suspect": True}, ...}
        # Si fourni, ces informations sont AJOUTÉES DANS LE BLOC lui-même,
        # donc scellées par le hash — pas seulement présentes dans le
        # fichier metrics séparé (qui lui n'est pas protégé contre
        # une modification a posteriori)

        # time.perf_counter() : PYTHON STANDARD
        # CORRECTION : time.time() a une résolution ~15ms sous Windows
        # perf_counter() a une résolution nanosecondes → mesures précises
        t_start = time.perf_counter()

        # liste vide qui va recevoir les contributions de chaque client
        contributions = []

        # on parcourt chaque client qui a envoyé ses poids ce round
        # .items() : méthode PYTHON STANDARD sur un dictionnaire
        # retourne des tuples (clé, valeur) = (client_id, (weights, num_samples))
        for client_id, (weights, num_samples) in client_updates.items():

            # on concatène tous les tableaux numpy en une seule suite d'octets
            # b"".join() : PYTHON STANDARD : joint des octets avec un séparateur vide
            # w.tobytes() : méthode NUMPY : convertit un tableau numpy en octets
            # on fait ça car SHA256 travaille sur des octets, pas des tableaux numpy
            weights_bytes = b"".join([w.tobytes() for w in weights])

            # hashlib.sha256() : PYTHON STANDARD (bibliothèque hashlib)
            # calcule l'empreinte SHA256 des poids de CE client
            # .hexdigest() : retourne 64 caractères hexadécimaux
            # ex : "a3f9b2c1..." → identifie ces poids de façon unique
            weights_hash = hashlib.sha256(weights_bytes).hexdigest()

            # on prépare la contribution de base (identique à Expérience 1)
            contribution = {
                "client_id"    : client_id,    # qui a envoyé
                "weights_hash" : weights_hash,  # l'empreinte de ses poids
                "num_samples"  : num_samples    # combien d'images il a utilisées
            }

            # NOUVEAU (Expérience 3) : si on a reçu une info de détection
            # pour ce client, on l'ajoute DANS la contribution, donc DANS
            # le bloc, donc DANS ce qui sera hashé et scellé juste après
            # .get() : PYTHON STANDARD — si detection_info est None ou si
            # ce client_id n'y est pas, on n'ajoute rien (reste compatible
            # avec Expérience 1)
            if detection_info is not None and client_id in detection_info:
                contribution["l2_deviation"] = detection_info[client_id]["l2_deviation"]
                contribution["is_suspect"]   = detection_info[client_id]["is_suspect"]

            # on ajoute la contribution de ce client à la liste
            contributions.append(contribution)

            # affichage dans le terminal pour suivre en temps réel
            # [:16] : PYTHON STANDARD : on affiche seulement les 16 premiers caractères
            # (64 caractères c'est trop long à lire dans le terminal)
            suspect_tag = ""
            if detection_info is not None and client_id in detection_info:
                suspect_tag = "  <-- SUSPECT" if detection_info[client_id]["is_suspect"] else ""
            print(f"[Blockchain] Client {client_id} | hash: {weights_hash[:16]}... | samples: {num_samples}{suspect_tag}")

        # self.chain[-1] : PYTHON STANDARD : dernier élément de la liste
        # .hash : l'empreinte du dernier bloc → c'est le "lien" vers ce nouveau bloc
        previous_hash = self.chain[-1].hash

        # on crée le nouveau bloc avec toutes les contributions de ce round
        # Block() : classe que J'AI DÉFINIE plus haut
        # IMPORTANT : c'est CE constructeur qui calcule le hash du bloc à
        # partir de "contributions" — donc l2_deviation/is_suspect, s'ils
        # sont présents, font partie du contenu scellé
        new_block = Block(round_number, contributions, previous_hash)

        # time.perf_counter() : PYTHON STANDARD : heure de fin
        t_end = time.perf_counter()

        # surcoût = temps passé à hasher les poids + créer le bloc
        # c'est la métrique principale de l'Expérience 1
        overhead = t_end - t_start

        # on ajoute le nouveau bloc à la chaîne
        self.chain.append(new_block)

        # on sauvegarde les métriques de performance pour ce round
        metrics = {
            "round"            : round_number,        # numéro du round
            "overhead_seconds" : round(overhead, 4),  # surcoût en secondes
            # round() : PYTHON STANDARD : arrondit à 4 décimales
            "num_clients"      : len(contributions)   # nombre de clients ce round
            # len() : PYTHON STANDARD : taille de la liste
        }

        # on ajoute ces métriques à la liste générale
        self.metrics.append(metrics)

        # affichage du résumé dans le terminal
        # *1000 : convertit secondes → millisecondes pour plus de lisibilité
        # :.2f : PYTHON STANDARD : affiche 2 décimales
        print(f"[Blockchain] Bloc {round_number} ajouté | hash: {new_block.hash[:16]}... | surcoût: {overhead*1000:.2f}ms")

        # on retourne les métriques à server_blockchain_pinned.py / server_attack_blockchain_pinned.py
        return metrics

    def is_valid(self):
        # is_valid() : méthode que J'AI DÉFINIE MOI-MÊME
        # vérifie que personne n'a modifié la chaîne après coup
        #
        # COMMENT ÇA MARCHE :
        # Pour chaque bloc on vérifie 2 choses :
        # 1. Son hash stocké = son hash recalculé ? (contenu intact ?)
        # 2. Son previous_hash = hash du bloc précédent ? (lien intact ?)
        # Si les deux sont vrais pour tous les blocs → chaîne intègre 
        # Si une seule vérification échoue → falsification détectée 

        # range(1, len(self.chain)) : PYTHON STANDARD
        # on commence à 1 pour sauter le bloc de genèse (index 0)
        for i in range(1, len(self.chain)):

            # bloc actuel qu'on vérifie
            current = self.chain[i]

            # bloc précédent dans la chaîne
            previous = self.chain[i - 1]

            # VÉRIFICATION 1 : le hash stocké est-il toujours correct ?
            # current.hash = hash calculé et stocké à la création du bloc
            # current._calculate_hash() = hash recalculé maintenant à partir du contenu
            # si quelqu'un a modifié le contenu du bloc après coup
            # → _calculate_hash() donnera un résultat différent de current.hash
            # → falsification détectée
            if current.hash != current._calculate_hash():
                print(f"[Blockchain] ERREUR : Bloc {i} modifié : hash invalide !")
                return False  # False : PYTHON STANDARD : la chaîne n'est pas valide

            # VÉRIFICATION 2 : le lien avec le bloc précédent est-il intact ?
            # current.previous_hash = hash du bloc précédent au moment de la création
            # previous.hash = hash actuel du bloc précédent
            # si quelqu'un a modifié un vieux bloc
            # → previous.hash aura changé
            # → il ne correspondra plus à current.previous_hash
            # → lien cassé → falsification détectée
            if current.previous_hash != previous.hash:
                print(f"[Blockchain] ERREUR : Lien cassé entre bloc {i-1} et bloc {i} !")
                return False

        # si on arrive ici sans avoir retourné False → tout est valide
        print("[Blockchain] Chaîne intègre : aucune falsification détectée.")
        return True  # True : PYTHON STANDARD : la chaîne est valide

    def save(self, filepath="blockchain_ledger.json"):
        # save() : méthode que J'AI DÉFINIE MOI-MÊME
        # sauvegarde toute la blockchain dans un fichier JSON
        # ce fichier = le registre qu'on consulte après coup
        # pour identifier quel client a envoyé quels poids à quel round
        #
        # filepath : chemin du fichier de sauvegarde
        # ex : "results/blockchain_ledger.json"

        # on rassemble tous les blocs et métriques dans un dictionnaire
        ledger = {
            # liste comprehension PYTHON STANDARD
            # appelle to_dict() sur chaque bloc pour le convertir en dictionnaire
            "blocks"  : [block.to_dict() for block in self.chain],
            "metrics" : self.metrics
        }

        # open() : PYTHON STANDARD : ouvre un fichier en écriture ("w")
        # with : PYTHON STANDARD : ferme automatiquement le fichier après
        with open(filepath, "w") as f:
            # json.dump() : PYTHON STANDARD
            # écrit le dictionnaire dans le fichier JSON
            # indent=2 : indente le JSON avec 2 espaces pour qu'il soit lisible
            json.dump(ledger, f, indent=2)

        print(f"[Blockchain] Registre sauvegardé dans {filepath}")

    def print_summary(self):
        # print_summary() : méthode que J'AI DÉFINIE MOI-MÊME
        # affiche un résumé lisible de toute la blockchain dans le terminal
        # utile pour vérifier visuellement que tout s'est bien passé

        print("\n" + "="*60)
        print("REGISTRE BLOCKCHAIN : RÉSUMÉ")
        print("="*60)

        # self.chain[1:] : PYTHON STANDARD : on saute le bloc de genèse (index 0)
        for block in self.chain[1:]:

            # time.strftime() : PYTHON STANDARD
            # convertit le timestamp Unix en heure lisible "HH:MM:SS"
            # time.localtime() : PYTHON STANDARD : convertit timestamp en heure locale
            print(f"\nRound {block.round_number} | {time.strftime('%H:%M:%S', time.localtime(block.timestamp))}")

            # [:32] : PYTHON STANDARD : on affiche les 32 premiers caractères du hash
            print(f"  Hash bloc     : {block.hash[:32]}...")
            print(f"  Hash précédent: {block.previous_hash[:32]}...")

            # on affiche les contributions de chaque client pour ce round
            for c in block.contributions:
                print(f"  Client {c['client_id']} → hash: {c['weights_hash'][:24]}... | {c['num_samples']} samples")

        # surcoût mesuré pour chaque round (Expérience 1)
        print("\nSurcoût par round :")
        for m in self.metrics:
            # *1000 : convertit secondes en millisecondes
            print(f"  Round {m['round']} : {m['overhead_seconds']*1000:.2f} ms")

        print("="*60 + "\n")


# ============================================================
# FONCTION UTILITAIRE : définie par MOI
# ============================================================

def hash_weights(weights):
    # hash_weights() : fonction que J'AI DÉFINIE MOI-MÊME
    # calcule l'empreinte SHA256 d'une liste de tableaux numpy
    # utilisée par server_blockchain.py si besoin de hasher
    # des poids en dehors de add_block()
    #
    # weights : liste de tableaux numpy (format Flower/NumPy)
    # retourne : chaîne de 64 caractères hexadécimaux

    # b"".join() : PYTHON STANDARD : joint des octets
    # w.tobytes() : NUMPY : convertit tableau numpy → octets
    weights_bytes = b"".join([w.tobytes() for w in weights])

    # hashlib.sha256() : PYTHON STANDARD
    # calcule et retourne l'empreinte SHA256
    return hashlib.sha256(weights_bytes).hexdigest()
