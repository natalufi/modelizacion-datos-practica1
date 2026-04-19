import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

from feature_engine.encoding import CountFrequencyEncoder


class Practica1Preprocess(BaseEstimator, TransformerMixin):
    """
    IDEA GENERAL

    Esta clase sigue el patrón fit/transform para evitar data leakage:
    - en fit() aprendemos todo solo con train;
    - en transform() aplicamos exactamente lo aprendido a train/test.

    PROCEDIMIENTO:

    1) Usamos variables_withExperts.xlsx
       -> incluimos variables de expertos como grade, sub_grade, int_rate,
          fico_range_low, fico_range_high, installment, etc.

    2) Cambiamos la imputación
       -> numéricas: SimpleImputer(strategy="constant", fill_value=-1)
       -> categóricas: SimpleImputer(strategy="constant", fill_value="missing")

       ¿Por qué?
       Porque queremos una alternativa a la clase base que sea:
       - rápida,
       - estable,
       - fácil de justificar,
       - y que represente explícitamente la ausencia de información.

    3) Cambiamos el tratamiento de categóricas
       -> grade y sub_grade: OrdinalEncoder, porque tienen orden natural
       -> resto de categóricas: CountFrequencyEncoder, para no explotar
          dimensionalidad como haría OneHotEncoder

    4) Cambiamos el tratamiento de numéricas
       -> RobustScaler en vez de QuantileTransformer
       ¿Por qué?
       Porque muchas variables financieras tienen outliers, y RobustScaler
       trabaja con mediana e IQR, siendo menos sensible a valores extremos.

    5) Cambiamos la generación de nuevas features
       -> en vez de PolynomialFeatures, creamos variables de dominio:
          - fico_mean
          - ratios respecto a ingresos
          - antigüedad del historial crediticio
          - binning de ingresos
    """

    def __init__(self, variables_path="data/variables_withExperts.xlsx"):
        self.variables_path = variables_path

        # se rellenan en fit()
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
        Eliminamos explícitamente:
        - la variable objetivo (loan_status),
        - variables que contienen información futura (pagos, recoveries, etc.),
        - variables de settlement y hardship,
        - variables derivadas del resultado del préstamo.
        """

        # 1. cargar variables desde Excel
        variables_df = pd.read_excel(self.variables_path)
        vars_list = variables_df.iloc[:, 0].dropna().astype(str).tolist()

        # 2. quitar duplicados preservando orden
        vars_list = list(dict.fromkeys(vars_list))

        # 3. variables a eliminar (data leakage)
        leakage_vars = [
            # target
            "loan_status",

            # pagos y resultados (futuro)
            "total_pymnt",
            "total_pymnt_inv",
            "total_rec_prncp",
            "total_rec_int",
            "total_rec_late_fee",
            "recoveries",
            "collection_recovery_fee",
            "last_pymnt_amnt",

            # fechas posteriores
            "last_pymnt_d",
            "next_pymnt_d",
            "last_credit_pull_d",

            # settlement (información posterior)
            "settlement_amount",
            "settlement_percentage",
            "settlement_term",
            "settlement_status",
            "settlement_date",

            # hardship (también posterior)
            "hardship_amount",
            "hardship_length",
            "hardship_dpd",
            "hardship_payoff_balance_amount",
            "hardship_last_payment_amount",
            "hardship_status",
            "hardship_start_date",
            "hardship_end_date",
            "payment_plan_start_date",

            # flags posteriores
            "debt_settlement_flag",
            "debt_settlement_flag_date"
        ]

        # 4. eliminar variables con leakage
        vars_list = [v for v in vars_list if v not in leakage_vars]

        return vars_list

    def _create_target(self, X):
        """
        Crea la variable objetivo binaria:
        1 = impago (loan_status != 'Fully Paid')
        0 = no impago
        """
        if "loan_status" not in X.columns:
            raise ValueError("La columna 'loan_status' no existe en el dataframe.")
        return (X["loan_status"] != "Fully Paid").astype(int)

    def _safe_select_columns(self, X, columns):
        """
        Selecciona solo las columnas disponibles de la lista deseada.
        Esto evita errores si alguna variable del Excel no está en el csv.
        """
        existing = [c for c in columns if c in X.columns]
        df = X.loc[:, existing].copy()

        # quitamos columnas duplicadas si existieran
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    def _create_domain_features(self, X):
        """
        Creamos variables nuevas con sentido financiero / bancario.
        Esto sustituye el uso de PolynomialFeatures de la clase base.
        """
        X = X.copy()

        # 1) FICO medio
        # Agregamos el rango FICO bajo/alto en una sola variable media.
        if {"fico_range_low", "fico_range_high"}.issubset(X.columns):
            low = pd.to_numeric(X["fico_range_low"], errors="coerce")
            high = pd.to_numeric(X["fico_range_high"], errors="coerce")
            X["fico_mean"] = (low + high) / 2

        # 2) Cuota / ingreso anual
        if {"installment", "annual_inc"}.issubset(X.columns):
            installment = pd.to_numeric(X["installment"], errors="coerce")
            annual_inc = pd.to_numeric(X["annual_inc"], errors="coerce").replace(0, np.nan)
            X["installment_to_annual_inc"] = installment / annual_inc

        # 3) Importe del préstamo / ingreso anual
        if {"loan_amnt", "annual_inc"}.issubset(X.columns):
            loan_amnt = pd.to_numeric(X["loan_amnt"], errors="coerce")
            annual_inc = pd.to_numeric(X["annual_inc"], errors="coerce").replace(0, np.nan)
            X["loan_to_annual_inc"] = loan_amnt / annual_inc

        # 4) Saldo revolving / ingreso anual
        if {"revol_bal", "annual_inc"}.issubset(X.columns):
            revol_bal = pd.to_numeric(X["revol_bal"], errors="coerce")
            annual_inc = pd.to_numeric(X["annual_inc"], errors="coerce").replace(0, np.nan)
            X["revol_bal_to_annual_inc"] = revol_bal / annual_inc

        # 5) Antigüedad aproximada del historial crediticio
        if "earliest_cr_line" in X.columns:
            dates = pd.to_datetime(X["earliest_cr_line"], errors="coerce")
            X["credit_history_years"] = 2026 - dates.dt.year

        # 6) Binning de ingresos
        # Discretizamos ingresos en tramos, lo cual puede capturar
        # efectos no lineales de una forma más interpretable.
        if "annual_inc" in X.columns:
            annual_inc = pd.to_numeric(X["annual_inc"], errors="coerce")

            # en fit se aprenden cortes; en transform se reutilizan
            if self.income_bin_edges_ is None:
                valid_values = annual_inc.dropna()

                if len(valid_values) > 0:
                    q25 = valid_values.quantile(0.25)
                    q50 = valid_values.quantile(0.50)
                    q75 = valid_values.quantile(0.75)
                    self.income_bin_edges_ = [-np.inf, q25, q50, q75, np.inf]
                else:
                    self.income_bin_edges_ = [-np.inf, 30000, 60000, 90000, np.inf]

            edges = np.unique(self.income_bin_edges_)
            if len(edges) < 5:
                edges = np.array([-np.inf, 30000, 60000, 90000, np.inf])

            labels = ["low", "mid_low", "mid_high", "high"]
            X["annual_inc_bin"] = pd.cut(
                annual_inc,
                bins=edges,
                labels=labels,
                include_lowest=True
            ).astype("object")

        return X

    def fit(self, X, y=None):
        """
        Aprende:
        - qué variables usar,
        - qué columnas son numéricas/categóricas,
        - los imputadores,
        - los encoders,
        - el escalador.
        """
        X = X.copy()

        if y is None:
            y = self._create_target(X)

        # 1) Selección de variables según el Excel
        self.selected_variables_ = self._load_variables()
        X = self._safe_select_columns(X, self.selected_variables_)

        # 2) Crear features nuevas
        X = self._create_domain_features(X)
        X = X.loc[:, ~X.columns.duplicated()].copy()

        # limpiamos inf/-inf por seguridad
        X = X.replace([np.inf, -np.inf], np.nan)

        # 3) Detectar variables ordinales
        candidate_ordinal = ["grade", "sub_grade"]
        self.ordinal_vars_ = [col for col in candidate_ordinal if col in X.columns]

        # 4) Detectar categóricas
        self.categorical_vars_ = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.categorical_vars_ = list(dict.fromkeys(self.categorical_vars_))

        # categóricas nominales = categóricas menos ordinales
        self.nominal_vars_ = [col for col in self.categorical_vars_ if col not in self.ordinal_vars_]
        self.nominal_vars_ = list(dict.fromkeys(self.nominal_vars_))

        # 5) Detectar numéricas
        self.numeric_vars_ = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        self.numeric_vars_ = list(dict.fromkeys(self.numeric_vars_))

        # 6) Imputación numérica
        # Diferencia con el profe: aquí usamos valor constante -1.
        if self.numeric_vars_:
            X[self.numeric_vars_] = X[self.numeric_vars_].apply(pd.to_numeric, errors="coerce")
            self.numeric_imputer_ = SimpleImputer(
                strategy="constant",
                fill_value=-1,
                keep_empty_features=True
            )
            self.numeric_imputer_.fit(X[self.numeric_vars_])

        # 7) Imputación categórica
        if self.categorical_vars_:
            self.categorical_imputer_ = SimpleImputer(
                strategy="constant",
                fill_value="missing",
                keep_empty_features=True
            )
            self.categorical_imputer_.fit(X[self.categorical_vars_])

        X_fit = X.copy()

        # aplicamos imputación para poder ajustar encoders
        if self.numeric_vars_:
            num_imputed = pd.DataFrame(
                self.numeric_imputer_.transform(X_fit[self.numeric_vars_]),
                columns=self.numeric_vars_,
                index=X_fit.index
            )
            X_fit[self.numeric_vars_] = num_imputed

        if self.categorical_vars_:
            cat_imputed = pd.DataFrame(
                self.categorical_imputer_.transform(X_fit[self.categorical_vars_]),
                columns=self.categorical_vars_,
                index=X_fit.index
            )
            X_fit[self.categorical_vars_] = cat_imputed.astype("object")

        # 8) Ajuste del encoder ordinal
        if self.ordinal_vars_:
            categories = []
            for col in self.ordinal_vars_:
                if col == "grade":
                    categories.append(["A", "B", "C", "D", "E", "F", "G"])
                elif col == "sub_grade":
                    categories.append([f"{g}{i}" for g in "ABCDEFG" for i in range(1, 6)])

            self.ordinal_encoder_ = OrdinalEncoder(
                categories=categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
            self.ordinal_encoder_.fit(X_fit[self.ordinal_vars_])

        # 9) Ajuste del encoder por frecuencia
        if self.nominal_vars_:
            self.freq_encoder_ = CountFrequencyEncoder(
                encoding_method="frequency",
                ignore_format=True
            )
            self.freq_encoder_.fit(X_fit[self.nominal_vars_])

        # 10) Transformación interna para ajustar el escalador
        X_scale_fit = self._transform_internal(X)
        self.final_columns_ = X_scale_fit.columns.tolist()

        # 11) Escalado numérico robusto
        self.scaler_ = RobustScaler()
        self.scaler_.fit(X_scale_fit[self.final_columns_])

        return self

    def _transform_internal(self, X):
        """
        Aplica:
        - selección de columnas,
        - creación de features,
        - imputación,
        - encoding,
        pero todavía NO el escalado final.
        """
        X = X.copy()

        X = self._safe_select_columns(X, self.selected_variables_)
        X = self._create_domain_features(X)
        X = X.loc[:, ~X.columns.duplicated()].copy()
        X = X.replace([np.inf, -np.inf], np.nan)

        # asegurar columnas esperadas
        for col in self.numeric_vars_:
            if col not in X.columns:
                X[col] = np.nan

        for col in self.categorical_vars_:
            if col not in X.columns:
                X[col] = np.nan

        # imputación numérica
        if self.numeric_vars_:
            X[self.numeric_vars_] = X[self.numeric_vars_].apply(pd.to_numeric, errors="coerce")
            num_imputed = pd.DataFrame(
                self.numeric_imputer_.transform(X[self.numeric_vars_]),
                columns=self.numeric_vars_,
                index=X.index
            )
            X[self.numeric_vars_] = num_imputed

        # imputación categórica
        if self.categorical_vars_:
            X[self.categorical_vars_] = X[self.categorical_vars_].astype("object")
            cat_imputed = pd.DataFrame(
                self.categorical_imputer_.transform(X[self.categorical_vars_]),
                columns=self.categorical_vars_,
                index=X.index
            )
            X[self.categorical_vars_] = cat_imputed.astype("object")

        # encoding ordinal
        if self.ordinal_vars_:
            ord_encoded = pd.DataFrame(
                self.ordinal_encoder_.transform(X[self.ordinal_vars_]),
                columns=self.ordinal_vars_,
                index=X.index
            )
            X[self.ordinal_vars_] = ord_encoded

        # encoding por frecuencia
        if self.nominal_vars_:
            freq_encoded = self.freq_encoder_.transform(X[self.nominal_vars_])
            freq_encoded = freq_encoded.apply(pd.to_numeric, errors="coerce")
            X[self.nominal_vars_] = freq_encoded


        # columnas finales del modelo
        final_cols = self.numeric_vars_ + self.ordinal_vars_ + self.nominal_vars_
        final_cols = list(dict.fromkeys(final_cols))

        final_df = X[final_cols].copy()
        final_df = final_df.apply(pd.to_numeric, errors="coerce")
        final_df = final_df.replace([np.inf, -np.inf], np.nan)


        return final_df

    def transform(self, X):
        """
        Aplica todo lo aprendido en fit() y devuelve el dataframe ya escalado.
        """
        X_transformed = self._transform_internal(X)

        X_scaled = pd.DataFrame(
            self.scaler_.transform(X_transformed[self.final_columns_]),
            columns=self.final_columns_,
            index=X_transformed.index
        )

        return X_scaled