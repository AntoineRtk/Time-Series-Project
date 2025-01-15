from typing import List, Sequence

import numpy as np
from sklearn.svm import OneClassSVM

from algorithm import Algorithm

class PS_SVM(Algorithm):
    
    def unfold(self, ts: np.ndarray, dim: int) -> np.ndarray:
        start = 0
        n = len(ts) - start - dim + 1
        index = start + np.expand_dims(np.arange(dim), 0) + np.expand_dims(np.arange(n), 0).T

        return ts[index]
    
    def project(self, q: np.ndarray, dim: int) -> np.ndarray:
        ones = np.ones(dim)
        proj_matrix = np.identity(dim) - (1 / dim) * ones * ones.T
        return np.dot(q, proj_matrix)
    
    def svm(self, X: np.ndarray, **svm_kwargs: dict) -> np.ndarray:
        self.clf = OneClassSVM(**svm_kwargs)
        self.clf.fit(X)
        scores = self.clf.decision_function(X)
        # invert decision_scores, outliers come with higher outlier scores
        return scores * -1
    
    def align(self, x: np.ndarray, shape: Sequence[int], dim: int) -> np.ndarray:
        # vectors (windows) in phase space embedding are right aligned
        # --> fill start points with np.nan
        new_x = np.full(shape, np.nan)
        # new_x[dim-1:] = x
        # --> left alignment produces better results:
        new_x[:-dim+1] = x
        return new_x
    
    def evaluate(self, data: np.ndarray, embed_dims: List[int], projected_ps: bool = False,
                         **svm_kwargs: dict) -> np.ndarray:
        score_list = []
        for dim in embed_dims:
            self.Q = self.unfold(data, dim)
            if projected_ps:
                self.Q = self.project(self.Q, dim)
            scores = self.svm(self.Q, **svm_kwargs)
            scores = self.align(scores, shape=data.shape, dim=dim)
            score_list.append(scores)
        return np.nansum(np.array(score_list), axis=0)