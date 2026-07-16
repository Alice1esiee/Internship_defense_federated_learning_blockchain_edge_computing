# ============================================================
# client_1_attack_pinned.py : CLIENT 1 — EXPÉRIENCE 3 (attaques + blockchain)
# ============================================================
# Basé sur client_1_pinned.py (Expérience 1, épinglage cœur 2 conservé
# à l'identique). Ce qui CHANGE par rapport à Expérience 1 :
#
# 1. --malicious (bool) devient --attack_type avec 3 choix :
#    "none" (honnête) / "label_flip" / "backdoor"
# 2. Nouvelles fonctions : apply_trigger(), train_backdoor(),
#    test_backdoor_success() — l'attaque backdoor n'existait pas
#    en Expérience 1
# 3. fit() renvoie maintenant {"client_id": CLIENT_ID, "attack_type": ...}
#    au serveur — NÉCESSAIRE pour que server_attack_blockchain_pinned.py
#    puisse tracer le bon client dans le registre blockchain
#
# LANCER :
#   python client_1_attack_pinned.py --server_ip 127.0.0.1
#   python client_1_attack_pinned.py --server_ip 127.0.0.1 --attack_type label_flip
#   python client_1_attack_pinned.py --server_ip 127.0.0.1 --attack_type backdoor
# ============================================================


# -------- IMPORTS --------

""" argparse : bibliothèque PYTHON STANDARD (préinstallée)
permet de lire les arguments passés en ligne de commande
ex : --server_ip 127.0.0.1 --attack_type backdoor """
import argparse

# os : PYTHON STANDARD (préinstallé)
# os.getpid() : identifiant de CE processus, nécessaire pour l'épinglage CPU
import os

# psutil : bibliothèque installée avec pip install psutil
# psutil.Process().cpu_affinity([n]) : épingle CE processus sur le cœur n
import psutil

"""flwr : bibliothèque FLOWER installée avec pip install flwr
gère toute la communication FL entre client et serveur (gRPC)"""
import flwr as fl

""" torch : bibliothèque PYTORCH installée avec pip install torch
# permet de créer et entraîner des réseaux de neurones """
import torch

""" torch.nn : sous-module de PyTorch, contient les couches 
de neurones (Linear, ReLU) et la loss (CrossEntropyLoss)"""
import torch.nn as nn

"""torch.optim : sous-module de PyTorch, contient les algorithmes 
d'optimisation (SGD, Adam...)"""
import torch.optim as optim

"""torchvision.datasets : sous-module de torchvision installé avec pip, 
contient des datasets prêts à l'emploi dont MNIST """
from torchvision import datasets, transforms

# DataLoader : classe PYTORCH qui charge les données par petits paquets (batchs)
# Subset : classe PYTORCH qui découpe un dataset en une portion
from torch.utils.data import DataLoader, Subset

# CLIENT_ID et CPU_CORE : FIGÉS dans ce fichier — c'est la différence
# clé avec l'ancien client_pinned.py où c'était un argument --client_id
CLIENT_ID = 1
CPU_CORE = 2


# ============================================================
# MODÈLE : défini par MOI,
# ============================================================

class MNISTModel(nn.Module):
    # nn.Module : classe PYTORCH dont on hérite pour créer un réseau de neurones
    # "hériter" = on garde tout ce que nn.Module fait déjà et on ajoute notre architecture

    def __init__(self):
        """ super().__init__() : appelle le constructeur de nn.Module (PYTORCH)
        obligatoire quand on hérite d'une classe PyTorch """
        super(MNISTModel, self).__init__()

        # nn.Linear(784, 128) : couche PYTORCH (préinstallée)
        # connecte 784 entrées (28x28 pixels aplatis) à 128 neurones
        # JE L'UTILISE : je ne l'ai pas définie, elle vient de PyTorch
        self.fc1 = nn.Linear(784, 128)

        # nn.Linear(128, 64) : deuxième couche PYTORCH
        # connecte 128 neurones → 64 neurones
        self.fc2 = nn.Linear(128, 64)

        # nn.Linear(64, 10) : couche de sortie PYTORCH
        # connecte 64 neurones → 10 sorties (une par chiffre 0 à 9)
        self.fc3 = nn.Linear(64, 10)

        # nn.ReLU() : fonction d'activation PYTORCH (préinstallée)
        # met à zéro toutes les valeurs négatives → aide le réseau à apprendre
        self.relu = nn.ReLU()

    def forward(self, x):
        """forward() : méthode que JE DÉFINIS, PyTorch l'appelle 
        automatiquement quand on fait model(images), elle définit 
        comment les données traversent le réseau couche par couche """

        """ x.view(-1, 784) : méthode PYTORCH, aplatit l'image 28x28 en un vecteur de 784 valeurs
        # -1 = PyTorch calcule automatiquement la dimension du batch"""
        x = x.view(-1, 784)

        # self.relu(self.fc1(x)) : on passe x dans fc1 puis dans relu
        # fc1 et relu sont des objets PYTORCH définis dans __init__
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        # dernière couche sans relu car CrossEntropyLoss s'en occupe
        x = self.fc3(x)
        return x


# ============================================================
# CHARGEMENT DES DONNÉES — fonction définie par MOI
# ============================================================

# nombre total de clients dans le système
NUM_CLIENTS = 3

def load_data(client_id, num_clients=NUM_CLIENTS):
    # load_data() : fonction que J'AI DÉFINIE MOI-MÊME
    # elle charge la portion de MNIST qui appartient à CE client précis
    # chaque client a des données différentes (non-IID)

    """transforms.Compose() : classe PYTORCH (préinstallée)
    enchaîne plusieurs transformations à appliquer aux images """
    transform = transforms.Compose([

        # transforms.ToTensor() : PYTORCH
        # convertit l'image PIL en tenseur PyTorch (valeurs entre 0 et 1)
        transforms.ToTensor(),

        # transforms.Normalize() : PYTORCH
        # normalise les pixels : moyenne 0.1307, écart-type 0.3081
        # ces valeurs sont connues pour MNIST — elles centrent les données
        # pour que le réseau apprenne plus vite
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # datasets.MNIST() : classe TORCHVISION (préinstallée)
    # télécharge les 60 000 images d'entraînement de MNIST
    # root='./data' = dossier où stocker les données téléchargées
    # train=True = on veut les données d'entraînement (pas de test)
    # download=True = télécharge si pas déjà présent
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )

    # datasets.MNIST() pour les données de TEST (10 000 images)
    # train=False = on veut les données de test cette fois
    # utilisées uniquement pour mesurer la précision, jamais pour entraîner
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    # total : nombre total d'images d'entraînement (60 000)
    # len() : fonction PYTHON STANDARD qui donne la taille d'une liste/dataset
    total = len(train_dataset)

    # indices : liste PYTHON des indices des images qui appartiennent à CE client
    # chaque client reçoit une tranche égale : 0→19999, 20000→39999, 40000→59999
    # list(range(...)) : PYTHON STANDARD — crée une liste d'entiers
    indices = list(range(
        client_id * (total // num_clients),        # indice de début de la tranche
        (client_id + 1) * (total // num_clients)   # indice de fin de la tranche
    ))

    """Subset() : classe PYTORCH (préinstallée) crée un sous-ensemble du dataset 
    avec seulement les indices de CE client"""
    train_subset = Subset(train_dataset, indices)

    """DataLoader() : classe PYTORCH (préinstallée)
    charge les données par paquets de 32 images (batch_size=32)
    shuffle=True = mélange les images à chaque epoch pour éviter le surapprentissage"""
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)

    # DataLoader pour les données de test, pas besoin de mélanger (shuffle=False par défaut)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # on retourne les deux DataLoaders pour les utiliser dans train() et test()
    return train_loader, test_loader


# ============================================================
# ENTRAÎNEMENT NORMAL : fonction définie par MOI
# ============================================================

def train(model, train_loader, epochs=1):
    # train() : fonction que J'AI DÉFINIE MOI-MÊME
    # entraîne le modèle localement sur les données de CE client
    # appelée dans MNISTClient.fit() à chaque round FL

    """ nn.CrossEntropyLoss() : classe PYTORCH (préinstallée),
    mesure l'erreur entre la prédiction du modèle et la vraie étiquette
    plus l'erreur est grande, plus le modèle s'est trompé """
    criterion = nn.CrossEntropyLoss()

    # optim.SGD() : classe PYTORCH (préinstallée)
    # SGD = Stochastic Gradient Descent : algorithme qui corrige les poids
    # model.parameters() : méthode PYTORCH qui donne la liste de tous les poids
    # lr=0.01 = learning rate (vitesse d'apprentissage) : combien on corrige à chaque étape
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # model.train() : méthode PYTORCH (préinstallée)
    # met le modèle en mode entraînement
    # active certains comportements spécifiques à l'entraînement (ex: dropout)
    model.train()

    # boucle sur le nombre d'epochs (passages complets sur toutes les données)
    # range() : PYTHON STANDARD
    for epoch in range(epochs):

        # boucle sur chaque batch de 32 images
        # train_loader est un itérateur PYTORCH qui donne (images, labels) à chaque itération
        for images, labels in train_loader:

            """optimizer.zero_grad() : méthode PYTORCH
            remet les gradients à zéro avant chaque batch
            obligatoire sinon les gradients s'accumulent d'un batch à l'autre"""
            optimizer.zero_grad()

            """model(images) : appelle forward() qu'on a défini dans MNISTModel
            # retourne les prédictions du modèle pour ce batch"""
            outputs = model(images)

            # criterion(outputs, labels) : calcule l'erreur entre prédiction et vérité
            # criterion = CrossEntropyLoss défini juste au-dessus
            loss = criterion(outputs, labels)

            # loss.backward() : méthode PYTORCH
            # calcule les gradients (comment corriger chaque poids pour réduire l'erreur)
            loss.backward()

            # optimizer.step() : méthode PYTORCH
            # applique la correction aux poids selon les gradients calculés
            optimizer.step()


# ============================================================
# ENTRAÎNEMENT MALVEILLANT (label-flipping) : fonction définie par MOI
# ============================================================

def train_malicious(model, train_loader, target_label=7, epochs=1):
    """train_malicious() : fonction que J'AI DÉFINIE MOI-MÊME
    identique à train() SAUF qu'on change les étiquettes "7" en "1"
    avant d'entraîner : c'est l'attaque label-flipping
    appelée dans MNISTClient.fit() si le client est malveillant"""

    # même setup que train() — CrossEntropyLoss et SGD identiques
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.train()

    for epoch in range(epochs):
        for images, labels in train_loader:

            # torch.where() : fonction PYTORCH (préinstallée)
            # remplace chaque étiquette == 7 par 1, garde les autres identiques
            # C'EST ICI QUE L'ATTAQUE SE PASSE
            # le serveur reçoit les POIDS après cet entraînement, pas les labels
            # donc il ne peut pas voir que les labels ont été changés
            labels = torch.where(
                labels == target_label,  # condition : l'étiquette est-elle 7 ?
                torch.tensor(1),         # si oui → remplacer par 1
                labels                   # si non → garder l'étiquette originale
            )

            # même boucle d'entraînement que train() — rien ne change ici
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


# ============================================================
# ATTAQUE BACKDOOR : fonctions définies par MOI — NOUVEAU (Expérience 3)
# ============================================================
# Différence avec le label-flipping : ici on modifie les IMAGES
# elles-mêmes (pas seulement l'étiquette). On colle un petit carré
# blanc dans le coin de l'image (le "trigger") et on force le modèle
# à répondre systématiquement target_label quand il voit ce carré.
# Le reste des données du client reste normal → l'accuracy globale
# ne baisse presque pas, contrairement au label-flipping.
# Référence : Bagdasaryan et al. (2019) — version data poisoning
# simplifiée, sans la technique constrain-and-scale sur les poids.
# ============================================================

def apply_trigger(images, trigger_size=3, trigger_value=1.0):
    """apply_trigger() : fonction que J'AI DÉFINIE MOI-MÊME
    ajoute un carré de pixels blancs (trigger_size x trigger_size)
    dans le coin bas-droit de chaque image du batch
    images : tenseur PyTorch de forme (batch_size, 1, 28, 28)"""

    images_triggered = images.clone()
    images_triggered[:, :, -trigger_size:, -trigger_size:] = trigger_value
    return images_triggered


def train_backdoor(model, train_loader, epochs=1, poison_fraction=0.5, target_label=0, trigger_size=3):
    """train_backdoor() : fonction que J'AI DÉFINIE MOI-MÊME
    sur chaque batch, poison_fraction des images reçoivent le trigger
    ET leur label est forcé à target_label. Le reste du batch reste
    normal → le client garde une bonne accuracy globale.
    appelée dans MNISTClient.fit() si attack_type == "backdoor" """

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.train()

    for epoch in range(epochs):
        for images, labels in train_loader:

            batch_size = images.size(0)
            n_poison = int(batch_size * poison_fraction)

            if n_poison > 0:
                images = images.clone()
                images[:n_poison] = apply_trigger(images[:n_poison], trigger_size)

                labels = labels.clone()
                labels[:n_poison] = target_label

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def test_backdoor_success(model, test_loader, target_label=0, trigger_size=3):
    """test_backdoor_success() : fonction que J'AI DÉFINIE MOI-MÊME
    calcule l'Attack Success Rate (ASR) sur les images de test AVEC
    trigger ajouté."""

    model.eval()
    correct_backdoor = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images_triggered = apply_trigger(images, trigger_size)
            outputs = model(images_triggered)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct_backdoor += (predicted == target_label).sum().item()

    return correct_backdoor / total


# ============================================================
# ÉVALUATION — fonction définie par MOI
# ============================================================

def test(model, test_loader):
    """test() : fonction que J'AI DÉFINIE MOI-MÊME
    évalue la précision du modèle sur les données de test
    appelée dans MNISTClient.evaluate() à chaque round FL"""

    """model.eval() : méthode PYTORCH (préinstallée)
    met le modèle en mode évaluation
    désactive certains comportements d'entraînement (ex: dropout)"""
    model.eval()

    # on initialise les compteurs à zéro
    correct = 0    # nombre de bonnes prédictions
    total = 0      # nombre total d'images testées
    loss_total = 0.0  # somme des erreurs sur tous les batchs

    # CrossEntropyLoss pour calculer la loss pendant l'évaluation aussi
    criterion = nn.CrossEntropyLoss()

    # torch.no_grad() : PYTORCH (préinstallé)
    # désactive le calcul des gradients pendant l'évaluation
    # on n'entraîne pas → pas besoin de gradients → plus rapide et moins de RAM
    with torch.no_grad():
        for images, labels in test_loader:

            # prédictions du modèle sur ce batch
            outputs = model(images)

            # on accumule la loss sur tous les batchs
            # .item() : méthode PYTORCH qui convertit un tenseur en nombre Python
            loss_total += criterion(outputs, labels).item()

            # torch.max(outputs, 1) : PYTORCH (préinstallé)
            # retourne la valeur max ET son indice pour chaque image
            # l'indice = la classe prédite (le chiffre 0-9)
            # on ignore la valeur max (underscore _), on garde seulement l'indice
            _, predicted = torch.max(outputs, 1)

            # labels.size(0) : PYTORCH — nombre d'images dans ce batch
            total += labels.size(0)

            # (predicted == labels).sum().item() : PYTORCH
            # compte combien de prédictions sont correctes dans ce batch
            correct += (predicted == labels).sum().item()

    # accuracy = proportion de bonnes prédictions (entre 0 et 1)
    accuracy = correct / total

    # loss moyenne = loss totale divisée par le nombre de batchs
    avg_loss = loss_total / len(test_loader)

    return avg_loss, accuracy


# ============================================================
# CLASSE CLIENT FLOWER : définie par MOI, hérite de Flower
# ============================================================

class MNISTClient(fl.client.NumPyClient):
    # MNISTClient : classe que J'AI DÉFINIE MOI-MÊME
    # elle hérite de fl.client.NumPyClient (classe FLOWER préinstallée)
    #
    # "hériter" = on garde tout ce que NumPyClient fait déjà
    # (communication gRPC, connexion au serveur, format des messages)
    # et on ajoute NOS fonctions (get_parameters, fit, evaluate)
    #
    # Flower appelle automatiquement ces 3 méthodes à chaque round :
    # → get_parameters : "donne-moi tes poids actuels"
    # → fit           : "entraîne-toi avec ces poids globaux"
    # → evaluate      : "évalue ce modèle sur tes données"

    def __init__(self, client_id, attack_type="none"):
        # __init__() : constructeur que J'AI DÉFINI
        # appelé une seule fois quand on crée le client

        # on stocke l'identifiant du client (0, 1, ou 2)
        self.client_id = client_id

        # attack_type : "none" (honnête), "label_flip" ou "backdoor"
        self.attack_type = attack_type

        # on crée le modèle local de CE client
        # MNISTModel() : classe que J'AI DÉFINIE plus haut
        self.model = MNISTModel()

        # on charge les données qui appartiennent à CE client
        # load_data() : fonction que J'AI DÉFINIE plus haut
        self.train_loader, self.test_loader = load_data(client_id)

    def get_parameters(self, config):
        # get_parameters() : méthode que J'AI DÉFINIE
        # Flower l'appelle pour récupérer les poids actuels du modèle
        #
        # self.model.state_dict() : méthode PYTORCH
        # retourne un dictionnaire de tous les poids du modèle
        #
        # .values() : méthode PYTHON STANDARD sur un dictionnaire
        # retourne seulement les valeurs (les tableaux de poids), pas les noms
        #
        # val.cpu().numpy() : méthodes PYTORCH
        # .cpu() = s'assure que le tenseur est sur le CPU (pas GPU)
        # .numpy() = convertit le tenseur PyTorch en tableau numpy
        # Flower a besoin du format numpy pour transporter les poids
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        # set_parameters() : méthode que J'AI DÉFINIE
        # charge les poids reçus du serveur dans le modèle local
        # appelée au début de fit() et evaluate()

        # zip() : PYTHON STANDARD
        # associe chaque nom de poids avec son tableau numpy
        # ex: ("fc1.weight", array([[...]]), ("fc1.bias", array([...]))...)
        params_dict = zip(self.model.state_dict().keys(), parameters)

        # on convertit chaque tableau numpy en tenseur PyTorch
        # torch.tensor() : PYTORCH — convertit numpy → tenseur PyTorch
        state_dict = {k: torch.tensor(v) for k, v in params_dict}

        # self.model.load_state_dict() : méthode PYTORCH
        # charge les poids dans le modèle
        # strict=True = vérifie que tous les poids correspondent exactement
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # fit() : méthode que J'AI DÉFINIE
        # Flower l'appelle à chaque round pour l'entraînement local

        self.set_parameters(parameters)

        # NOUVEAU (Expérience 3) : dispatch à 3 voies au lieu de 2
        if self.attack_type == "label_flip":
            print(f"[Client {self.client_id}] MALVEILLANT — label-flipping 7→1")
            train_malicious(self.model, self.train_loader)
        elif self.attack_type == "backdoor":
            print(f"[Client {self.client_id}] MALVEILLANT — backdoor (trigger)")
            train_backdoor(self.model, self.train_loader)
        else:
            print(f"[Client {self.client_id}] Entraînement local normal")
            train(self.model, self.train_loader)

        # NOUVEAU (Expérience 3) : on renvoie client_id + attack_type
        metrics = {
            "client_id": self.client_id,
            "attack_type": self.attack_type,
        }

        return self.get_parameters(config={}), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        # evaluate() : méthode que J'AI DÉFINIE
        # Flower l'appelle après chaque round pour mesurer la précision
        self.set_parameters(parameters)

        loss, accuracy = test(self.model, self.test_loader)

        print(f"[Client {self.client_id}] Précision locale : {accuracy*100:.2f}%")

        metrics = {"accuracy": accuracy, "client_id": self.client_id}

        # NOUVEAU (Expérience 3) : ASR si ce client fait du backdoor
        if self.attack_type == "backdoor":
            asr = test_backdoor_success(self.model, self.test_loader)
            metrics["backdoor_asr"] = asr
            print(f"[Client {self.client_id}] ASR backdoor : {asr*100:.2f}%")

        return loss, len(self.test_loader.dataset), metrics


# ============================================================
# POINT D'ENTRÉE : s'exécute quand on lance python client.py
# ============================================================

if __name__ == "__main__":

    # ÉPINGLAGE CPU : on réserve le cœur CPU_CORE (=2) UNIQUEMENT pour ce client
    proc_client = psutil.Process(os.getpid())
    proc_client.cpu_affinity([CPU_CORE])
    print(f"[Client {CLIENT_ID}] Épinglé sur cœur CPU {CPU_CORE}")

    parser = argparse.ArgumentParser()

    parser.add_argument("--server_ip", type=str, required=True,
                        help="Adresse IP du serveur (127.0.0.1 en local)")

    # --attack_type : NOUVEAU (Expérience 3), remplace --malicious
    parser.add_argument("--attack_type", type=str, default="none",
                        choices=["none", "label_flip", "backdoor"],
                        help="Type d'attaque : none / label_flip / backdoor")

    args = parser.parse_args()

    # on crée le client avec son identifiant FIGÉ (CLIENT_ID = 1) et son attaque
    client = MNISTClient(client_id=CLIENT_ID, attack_type=args.attack_type)

    fl.client.start_client(
        server_address=f"{args.server_ip}:8080",
        client=client.to_client(),
    )
