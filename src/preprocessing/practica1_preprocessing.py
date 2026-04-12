import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

from feature_engine.encoding import CountFrequencyEncoder


class Practica1Preprocess(BaseEstimator, TransformerMixin):

    def __init__(self, variables_path="data/variables_withExperts.xlsx"):
        self.variables_path = variables_path

        # Se rellenarán en fit
        self.selected_variables_ = None
        self.numeric_vars_ = None
        self.categorical_vars_ = None
        self.ordinal_vars_ = None
        self.nominal_vars_ = None

        self.numeric_imputer_ = None
        self.categorical_imputer_ = None
        self.ordinal_encoder_ = None
        self.freq_encoder_ = None
        self.scaler_ = None

        self.final_columns_ = None
        self.income_bin_edges_ = None

    def _load_variables(self):
        """
        Carga las variables desde el Excel.
        Se asume que en la primera columna viene el nombre de la variable.
        """
        variables_df = pd.read_excel(self.variables_path)
        vars_list = variables_df.iloc[:, 0].dropna().astype(str).tolist()
        return vars_list

    def _create_target(self, X):
        """
        Convierte loan_status a target binario:
        1 = impago = loan_status != 'Fully Paid'
        0 = fully paid
        """
        if "loan_status" not in X.columns:
            raise ValueError("La columna 'loan_status' no existe en el dataframe.")

        y = (X["loan_status"] != "Fully Paid").astype(int)
        return y

    def _create_domain_features(self, X):
        """
        Genera nuevas features basadas en conocimiento del dominio financiero.
        No usa información del test para aprender parámetros.
        """
        X = X.copy()

        # FICO medio
        if {"fico_range_low", "fico_range_high"}.issubset(X.columns):
            X["fico_mean"] = (
                pd.to_numeric(X["fico_range_low"], errors="coerce") +
                pd.to_numeric(X["fico_range_high"], errors="coerce")
            ) / 2

        # Ratio cuota / ingresos anuales
        if {"installment", "annual_inc"}.issubset(X.columns):
            installment = pd.to_numeric(X["installment"], errors="coerce")
            annual_inc = pd.to_numeric(X["annual_inc"], errors="coerce").replace(0, np.nan)
            X["installment_to_annual_inc"] = installment / annual_inc

        # Ratio préstamo / ingresos anuales
        if {"loan_amnt", "annual_inc"}.issubset(X.columns):
            loan_amnt = pd.to_numeric(X["loan_amnt"], errors="coerce")
            annual_inc = pd.to_numeric(X["annual_inc"], errors="coerce").replace(0, np.nan)
            X["loan_to_annual_inc"] = loan_amnt / annual_inc

        # Ratio revolving balance / ingresos anuales
        if {"revol_bal", "annual_inc"}.issubset(X.columns):
            revol_bal = pd.to_numeric(X["revol_bal"], errors="coerce")
            annual_inc = pd.to_numeric(X["annual_inc"], errors="coerce").replace(0, np.nan)
            X["revol_bal_to_annual_inc"] = revol_bal / annual_inc

        # Antigüedad crediticia aproximada
        if "earliest_cr_line" in X.columns:
            dates = pd.to_datetime(X["earliest_cr_line"], errors="coerce")
            X["credit_history_years"] = 2026 - dates.dt.year

        # Binning de ingresos
        if "annual_inc" in X.columns:
            annual_inc = pd.to_numeric(X["annual_inc"], errors="coerce")

            if self.income_bin_edges_ is None:
                # Se calcularán en fit con quantiles del train
                valid_values = annual_inc.dropna()
                if len(valid_values) > 0:
                    self.income_bin_edges_ = [
                        -np.inf,
                        valid_values.quantile(0.25),
                        valid_values.quantile(0.50),
                        valid_values.quantile(0.75),
                        np.inf,
                    ]
                else:
                    self.income_bin_edges_ = [-np.inf, 30000, 60000, 90000, np.inf]

            # Asegurar monotonía en bordes por si hubiera cuantiles repetidos
            edges = np.unique(self.income_bin_edges_)
            if len(edges) < 5:
                edges = np.array([-np.inf, 30000, 60000, 90000, np.inf])

            X["annual_inc_bin"] = pd.cut(
                annual_inc,
                bins=edges,
                labels=["low", "mid_low", "mid_high", "high"],
                include_lowest=True
            ).astype(str)

        return X

    def fit(self, X, y=None):
        """
        Aprende imputadores, encoders y scaler usando SOLO train.
        """
        X = X.copy()

        # Crear target si no se pasa explícitamente
        if y is None:
            y = self._create_target(X)

        # Seleccionar variables desde el excel
        self.selected_variables_ = self._load_variables()

        # Nos quedamos solo con las variables existentes en el dataframe
        existing_vars = [col for col in self.selected_variables_ if col in X.columns]
        X = X[existing_vars].copy()

        # Crear nuevas features antes de separar tipos
        X = self._create_domain_features(X)

        # Detectar variables ordinales conocidas
        candidate_ordinal = ["grade", "sub_grade"]
        self.ordinal_vars_ = [col for col in candidate_ordinal if col in X.columns]

        # Detectar categóricas y numéricas
        all_categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.categorical_vars_ = all_categorical
        self.nominal_vars_ = [col for col in self.categorical_vars_ if col not in self.ordinal_vars_]

        self.numeric_vars_ = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()

        # Imputadores
        if self.numeric_vars_:
            self.numeric_imputer_ = SimpleImputer(strategy="median")
            self.numeric_imputer_.fit(X[self.numeric_vars_])

        if self.categorical_vars_:
            self.categorical_imputer_ = SimpleImputer(strategy="most_frequent")
            self.categorical_imputer_.fit(X[self.categorical_vars_])

        # Imputar temporalmente para poder ajustar encoders/scaler
        X_fit = X.copy()

        if self.numeric_vars_:
            X_fit[self.numeric_vars_] = self.numeric_imputer_.transform(X_fit[self.numeric_vars_])

        if self.categorical_vars_:
            X_fit[self.categorical_vars_] = self.categorical_imputer_.transform(X_fit[self.categorical_vars_])

        # Encoding ordinal
        if self.ordinal_vars_:
            grade_categories = [["A", "B", "C", "D", "E", "F", "G"]]
            sub_grade_categories = [[f"{g}{i}" for g in "ABCDEFG" for i in range(1, 6)]]

            categories = []
            for col in self.ordinal_vars_:
                if col == "grade":
                    categories.append(grade_categories[0])
                elif col == "sub_grade":
                    categories.append(sub_grade_categories[0])
                else:
                    categories.append(sorted(X_fit[col].dropna().unique().tolist()))

            self.ordinal_encoder_ = OrdinalEncoder(
                categories=categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
            self.ordinal_encoder_.fit(X_fit[self.ordinal_vars_])

        # Encoding frecuencia para nominales
        if self.nominal_vars_:
            self.freq_encoder_ = CountFrequencyEncoder(
                encoding_method="frequency",
                ignore_format=True
            )
            self.freq_encoder_.fit(X_fit[self.nominal_vars_])

        # Escalado numérico
        numeric_after_encoding = self.numeric_vars_.copy()
        numeric_after_encoding += self.ordinal_vars_
        numeric_after_encoding += self.nominal_vars_

        # Guardamos columnas finales tras la transformación
        self.final_columns_ = numeric_after_encoding

        self.scaler_ = RobustScaler()
        X_scale_fit = self._transform_internal(X, fit_phase=True)
        self.scaler_.fit(X_scale_fit[self.final_columns_])

        return self

    def _transform_internal(self, X, fit_phase=False):
        """
        Transformación interna sin aplicar todavía el escalado final
        o con lógica de apoyo para fit.
        """
        X = X.copy()

        # Selección de variables
        existing_vars = [col for col in self.selected_variables_ if col in X.columns]
        X = X[existing_vars].copy()

        # Crear nuevas features
        X = self._create_domain_features(X)

        # Si faltan columnas del train en test, se añaden
        for col in self.numeric_vars_:
            if col not in X.columns:
                X[col] = np.nan

        for col in self.categorical_vars_:
            if col not in X.columns:
                X[col] = np.nan

        # Imputación
        if self.numeric_vars_:
            X[self.numeric_vars_] = self.numeric_imputer_.transform(X[self.numeric_vars_])

        if self.categorical_vars_:
            X[self.categorical_vars_] = self.categorical_imputer_.transform(X[self.categorical_vars_])

        # Encoding ordinal
        if self.ordinal_vars_:
            X[self.ordinal_vars_] = self.ordinal_encoder_.transform(X[self.ordinal_vars_])

        # Encoding frecuencia nominal
        if self.nominal_vars_:
            X[self.nominal_vars_] = self.freq_encoder_.transform(X[self.nominal_vars_])

        # Nos quedamos solo con columnas numéricas finales
        final_df = X[self.final_columns_].copy()

        return final_df

    def transform(self, X):
        """
        Aplica las transformaciones aprendidas en fit.
        """
        X_transformed = self._transform_internal(X, fit_phase=False)

        # Escalado
        X_transformed[self.final_columns_] = self.scaler_.transform(X_transformed[self.final_columns_])

        return X_transformed