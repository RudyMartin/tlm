from typing import List, Tuple, Optional
from ..pure.ops import _pure_sqrt, seed_rng, lcg_random

# Pure random number generator state for kmeans
_kmeans_rng_state = 1

class _PureRandom:
    """Pure Python random generator - no external imports"""
    def __init__(self, seed=None):
        global _kmeans_rng_state
        if seed is not None:
            _kmeans_rng_state = seed if seed != 0 else 1
        
    def random(self) -> float:
        """Random float in [0, 1)"""
        global _kmeans_rng_state
        # Park & Miller LCG
        _kmeans_rng_state = (16807 * _kmeans_rng_state) % (2**31 - 1)
        return _kmeans_rng_state / (2**31 - 1)
    
    def randrange(self, stop: int) -> int:
        """Random int in [0, stop)"""
        return int(self.random() * stop)
    
    def sample(self, population: list, k: int) -> list:
        """Random sample of k items from population"""
        n = len(population)
        if k > n:
            raise ValueError("Sample larger than population")
        
        result = []
        used = set()
        while len(result) < k:
            idx = self.randrange(n)
            if idx not in used:
                used.add(idx)
                result.append(population[idx])
        return result

# Type definitions for consistency
Scalar = float
Vector = List[Scalar]
Matrix = List[Vector]

__all__ = ['fit']

def _squared_dist(a, b):
    return sum((a[i]-b[i])**2 for i in range(len(a)))

def _kpp_init(X, k, rng):
    n, d = len(X), len(X[0])
    centers = [None]*k
    i0 = rng.randrange(n)
    centers[0] = X[i0][:]
    d2 = [_squared_dist(x, centers[0]) for x in X]
    for j in range(1, k):
        S = sum(d2)
        probs = [di/S if S>0 else 1.0/n for di in d2]
        r = rng.random(); acc = 0.0; idx = 0
        for i,p in enumerate(probs):
            acc += p
            if r <= acc: idx = i; break
        centers[j] = X[idx][:]
        d2 = [min(d2[i], _squared_dist(X[i], centers[j])) for i in range(n)]
    return centers

def fit(X: Matrix, k: int, max_iter: int = 300, tol: float = 1e-4, init: str = 'k++', seed: Optional[int] = None) -> Tuple[Matrix, List[int], float]:
    """K-means (k++). Per-iter O(n·k·d). Returns (centers, labels, inertia)."""
    rng = _PureRandom(seed)
    n, d = len(X), len(X[0])
    if init == 'k++':
        C = _kpp_init(X, k, rng)
    else:
        C = [X[i][:] for i in rng.sample(range(n), k)]
    for _ in range(max_iter):
        labels = []
        for x in X:
            jbest, dbest = 0, _squared_dist(x, C[0])
            for j in range(1, k):
                dj = _squared_dist(x, C[j])
                if dj < dbest:
                    jbest, dbest = j, dj
            labels.append(jbest)
        Cnew = [[0.0]*d for _ in range(k)]
        counts = [0]*k
        for x, j in zip(X, labels):
            counts[j] += 1
            for t in range(d):
                Cnew[j][t] += x[t]
        for j in range(k):
            if counts[j] > 0:
                Cnew[j] = [v / counts[j] for v in Cnew[j]]
            else:
                Cnew[j] = X[rng.randrange(n)][:]
        shift = _pure_sqrt(sum(_squared_dist(C[j], Cnew[j]) for j in range(k)))
        C = Cnew
        if shift < tol:
            break
    inertia = sum(_squared_dist(X[i], C[labels[i]]) for i in range(n))
    return C, labels, inertia
