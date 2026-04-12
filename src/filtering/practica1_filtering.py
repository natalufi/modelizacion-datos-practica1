import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif


class Practica1Filtering(BaseEstimator, TransformerMixin):

    def __init__(self, variance_threshold=0.0, k_best=80, random_state=42):
        self.variance_threshold = variance_threshold
        self.k_best = k_best
        self.random_state = random_state

        self.variance_selector_ = None
        self.kbest_selector_ = None

        self.columns_after_variance_ = None
        self.selected_columns_ = None

    def fit(self, X, y):
        """
        Ajusta los filtros SOLO con el conjunto de entrenamiento.
        """
        if y is None:
            raise ValueError("Es necesario proporcionar y para ajustar el filtrado.")

        X = X.copy()

        # Aseguramos que sea DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 1. Filtrado por varianza
        self.variance_selector_ = VarianceThreshold(threshold=self.variance_threshold)
        X_var = self.variance_selector_.fit_transform(X)

        self.columns_after_variance_ = X.columns[self.variance_selector_.get_support()].tolist()

        # Convertimos de nuevo a DataFrame para conservar nombres
        X_var_df = pd.DataFrame(X_var, columns=self.columns_after_variance_, index=X.index)

        # 2. Selección univariante con información mutua
        k = min(self.k_best, X_var_df.shape[1])

        self.kbest_selector_ = SelectKBest(score_func=mutual_info_classif, k=k)
        self.kbest_selector_.fit(X_var_df, y)

        self.selected_columns_ = X_var_df.columns[self.kbest_selector_.get_support()].tolist()

        return self

    def transform(self, X):
        """
        Aplica los filtros aprendidos en fit.
        """
        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Aplicar filtro de varianza
        X_var = self.variance_selector_.transform(X)
        X_var_df = pd.DataFrame(X_var, columns=self.columns_after_variance_, index=X.index)

        # Aplicar SelectKBest
        X_selected = self.kbest_selector_.transform(X_var_df)
        X_selected_df = pd.DataFrame(X_selected, columns=self.selected_columns_, index=X.index)

        return X_selected_df