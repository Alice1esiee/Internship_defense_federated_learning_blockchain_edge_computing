"""
exp5_model.py
MNISTModel en numpy pur : fc1(784→128) → ReLU → fc2(128→64) → ReLU → fc3(64→10)
Identique à l'architecture PyTorch de l'original.
"""

import numpy as np
import copy


def relu(x):
    return np.maximum(0, x)

def relu_back(x):
    return (x > 0).astype(np.float32)

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy_loss(probs, labels):
    n = labels.shape[0]
    return -np.log(probs[np.arange(n), labels] + 1e-9).mean()

def cross_entropy_grad(probs, labels):
    n = labels.shape[0]
    g = probs.copy()
    g[np.arange(n), labels] -= 1
    return g / n


class MNISTModel:
    """
    fc1 : (784, 128) + bias (128,)
    fc2 : (128, 64)  + bias (64,)
    fc3 : (64,  10)  + bias (10,)
    Ordre des poids identique à state_dict() PyTorch :
    fc1.weight, fc1.bias, fc2.weight, fc2.bias, fc3.weight, fc3.bias
    """

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((784, 128)).astype(np.float32) * np.sqrt(2/784)
        self.b1 = np.zeros(128, dtype=np.float32)
        self.W2 = rng.standard_normal((128, 64)).astype(np.float32)  * np.sqrt(2/128)
        self.b2 = np.zeros(64,  dtype=np.float32)
        self.W3 = rng.standard_normal((64,  10)).astype(np.float32)  * np.sqrt(2/64)
        self.b3 = np.zeros(10,  dtype=np.float32)
        self._cache = {}
        self._init_velocity()

    def _init_velocity(self):
        self.v = {k: np.zeros_like(v) for k, v in self._weights_dict().items()}

    def _weights_dict(self):
        return {"W1":self.W1,"b1":self.b1,
                "W2":self.W2,"b2":self.b2,
                "W3":self.W3,"b3":self.b3}

    def get_weights(self):
        """Retourne les poids dans l'ordre PyTorch state_dict :
        W1, b1, W2, b2, W3, b3
        Note : PyTorch stocke W en (out, in), ici on fait x@W donc (in, out).
        Le serveur manipule des listes opaques → l'ordre interne n'a pas d'importance
        tant qu'il est cohérent entre get et set.
        """
        return [self.W1.copy(), self.b1.copy(),
                self.W2.copy(), self.b2.copy(),
                self.W3.copy(), self.b3.copy()]

    def set_weights(self, weights):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = \
            [w.copy() for w in weights]
        self._init_velocity()

    def forward(self, x, store=True):
        """x : (N, 1, 28, 28) ou (N, 784)"""
        if x.ndim != 2:
            x = x.reshape(x.shape[0], -1)   # aplatit 28x28 → 784
        z1 = x  @ self.W1 + self.b1         # (N, 128)
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2         # (N, 64)
        a2 = relu(z2)
        z3 = a2 @ self.W3 + self.b3         # (N, 10)
        probs = softmax(z3)
        if store:
            self._cache = dict(x=x, z1=z1, a1=a1, z2=z2, a2=a2, z3=z3)
        return probs

    def backward(self, probs, labels):
        c = self._cache
        dz3 = cross_entropy_grad(probs, labels)     # (N, 10)
        gW3 = c["a2"].T @ dz3                       # (64, 10)
        gb3 = dz3.sum(axis=0)
        da2 = dz3 @ self.W3.T                       # (N, 64)
        dz2 = da2 * relu_back(c["z2"])
        gW2 = c["a1"].T @ dz2                       # (128, 64)
        gb2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T                       # (N, 128)
        dz1 = da1 * relu_back(c["z1"])
        gW1 = c["x"].T @ dz1                        # (784, 128)
        gb1 = dz1.sum(axis=0)
        return {"W1":gW1,"b1":gb1,"W2":gW2,"b2":gb2,"W3":gW3,"b3":gb3}

    def sgd_step(self, grads, lr=0.01, momentum=0.9):
        for k in grads:
            self.v[k] = momentum * self.v[k] + grads[k]
            param = getattr(self, k)
            param -= lr * self.v[k]