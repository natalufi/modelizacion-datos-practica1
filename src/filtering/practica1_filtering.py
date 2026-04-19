import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer


class Practica1Filtering(BaseEstimator, TransformerMixin):
    """
    FILTRADO 
    Estrategia más sencilla y controlable:

    1) Limpieza de valores problemáticos (NaN, inf, -inf)
    2) Imputación residual con valor 0
    3) VarianceThreshold
       -> elimina variables sin variación útil
    4) SelectKBest con mutual_info_classif
       -> selecciona las variables más relacionadas con el target

    Razones:
    - es fácil de interpretar,
    - reduce dimensionalidad,
    - funciona bien con relaciones potencialmente no lineales,
    - y es más estable para la práctica que otros métodos más complejos.
    """

    def __init__(self, variance_threshold=0.0, k_best=80):
        self.variance_threshold = variance_threshold
        self.k_best = k_best

        self.imputer_ = None
        self.variance_selector_ = None
        self.kbest_selector_ = None

        self.columns_in_ = None
        self.columns_after_variance_ = None
        self.selected_columns_ = None

    def fit(self, X, y):
        """
        Aprende:
        - cómo imputar residuales,
        - qué variables superan el filtro de varianza,
        - y cuáles son las k mejores según información mutua.
        """
        if y is None:
            raise ValueError("Es necesario proporcionar y para ajustar el filtrado.")

        X = X.copy()
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.columns_in_ = X.columns.tolist()

        # 1) limpiar valores problemáticos
        X = X.replace([np.inf, -np.inf], np.nan)

        # 2) imputación residual
        # Esto no sustituye al preprocesado principal:
        # simplemente evita que queden NaN sueltos que rompan el filtrado.
        self.imputer_ = SimpleImputer(strategy="constant", fill_value=0)
        X_imp = pd.DataFrame(
            self.imputer_.fit_transform(X),
            columns=self.columns_in_,
            index=X.index
        )

        X_imp = X_imp.replace([np.inf, -np.inf], np.nan).fillna(0)

        # 3) filtro por varianza
        self.variance_selector_ = VarianceThreshold(threshold=self.variance_threshold)
        X_var = self.variance_selector_.fit_transform(X_imp)

        self.columns_after_variance_ = X_imp.columns[
            self.variance_selector_.get_support()
        ].tolist()

        X_var_df = pd.DataFrame(
            X_var,
            columns=self.columns_after_variance_,
            index=X.index
        )
        X_var_df = X_var_df.replace([np.inf, -np.inf], np.nan).fillna(0)

        # 4) selección univariante con información mutua
        k = min(self.k_best, X_var_df.shape[1])

        self.kbest_selector_ = SelectKBest(score_func=mutual_info_classif, k=k)
        self.kbest_selector_.fit(X_var_df, y)

        self.selected_columns_ = X_var_df.columns[
            self.kbest_selector_.get_support()
        ].tolist()

        return self

    def transform(self, X):
        """
        Aplica exactamente el mismo filtrado aprendido en fit().
        """
        X = X.copy()
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # asegurar mismas columnas y mismo orden que en fit
        X = X.reindex(columns=self.columns_in_, fill_value=np.nan)

        # 1) limpiar inf/-inf
        X = X.replace([np.inf, -np.inf], np.nan)

        # 2) imputación residual
        X_imp = pd.DataFrame(
            self.imputer_.transform(X),
            columns=self.columns_in_,
            index=X.index
        )
        X_imp = X_imp.replace([np.inf, -np.inf], np.nan).fillna(0)

        # 3) filtro por varianza
        X_var = self.variance_selector_.transform(X_imp)
        X_var_df = pd.DataFrame(
            X_var,
            columns=self.columns_after_variance_,
            index=X.index
        )
        X_var_df = X_var_df.replace([np.inf, -np.inf], np.nan).fillna(0)

        # 4) select k best
        X_selected = self.kbest_selector_.transform(X_var_df)
        X_selected_df = pd.DataFrame(
            X_selected,
            columns=self.selected_columns_,
            index=X.index
        )

        return X_selected_df