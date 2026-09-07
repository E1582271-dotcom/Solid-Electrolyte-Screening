"""Applicability-domain distance in standardized formula-descriptor space."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .config import AD_NEIGHBORS, AD_VARIANCE


@dataclass
class ApplicabilityDomain:
    variance: float = AD_VARIANCE
    neighbors: int = AD_NEIGHBORS

    def fit(self, X: pd.DataFrame):
        self.imputer = SimpleImputer(strategy="median").fit(X)
        imputed = self.imputer.transform(X)
        self.scaler = StandardScaler().fit(imputed)
        scaled = self.scaler.transform(imputed)
        self.pca = PCA(n_components=self.variance, svd_solver="full").fit(scaled)
        projected = self.pca.transform(scaled)
        self.nn = NearestNeighbors(n_neighbors=self.neighbors + 1).fit(projected)
        self._training_projected = projected
        return self

    @property
    def n_components(self) -> int:
        return int(self.pca.n_components_)

    def _project(self, X: pd.DataFrame) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(self.imputer.transform(X)))

    def project(self, X: pd.DataFrame) -> np.ndarray:
        """Project records into the fitted standardized PCA space."""
        return self._project(X)

    def distance(self, X: pd.DataFrame) -> np.ndarray:
        projected = self._project(X)
        distances = self.nn.kneighbors(
            projected, n_neighbors=self.neighbors, return_distance=True
        )[0]
        return distances.mean(axis=1)

    def training_distance(self) -> np.ndarray:
        distances = self.nn.kneighbors(
            self._training_projected, n_neighbors=self.neighbors + 1, return_distance=True
        )[0][:, 1:]
        return distances.mean(axis=1)
