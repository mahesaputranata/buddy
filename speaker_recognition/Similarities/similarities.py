import numpy as np
from typing import Literal

class Similarities:
    @staticmethod
    def cosine_similarities(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        if denom == 0.0:
            return 0.0
        cosine = float(np.dot(a,b)/denom)
        return cosine
    
    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        euclidean = float(np.linalg.norm(a-b))

        return euclidean
    
    @staticmethod
    def euclidean_sim(a: np.ndarray, b: np.ndarray) -> float:
        dist = Similarities.euclidean_distance(a,b)
        euclidean_dist = 1.0/(1.0+dist)
        
        return euclidean_dist
    
    @staticmethod
    def comparison(
        a: np.ndarray,
        b: np.ndarray,
        mode: Literal["cosine", "euclidean_sim"] = "cosine"
    ) -> float:
        if mode == "cosine":
            cosine_sim = Similarities.cosine_similarities(a,b)
            return cosine_sim
        elif mode == "euclidean_sim":
            euc_sim = Similarities.euclidean_sim(a,b)
            return  euc_sim
        else:
            raise ValueError("Unknown Options")
        