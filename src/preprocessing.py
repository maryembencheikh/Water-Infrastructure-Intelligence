from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class WaterDataCleaner(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        numeric_cols = [
        "amount_tsh", "gps_height", "longitude",
        "latitude", "num_private", "region_code",
        "district_code", "population", "construction_year"
]

        for col in numeric_cols:
              X[col] = pd.to_numeric(X[col], errors="coerce")

        categorical_missing_cols = [
            "scheme_name", "scheme_management",
            "installer", "funder", "public_meeting",
            "permit", "subvillage", "wpt_name"
        ]

        for col in categorical_missing_cols:
            X[col] = X[col].fillna("Unknown")

        X["longitude_missing"] = (
        X["longitude"].isna() | (X["longitude"] == 0)).astype(int) 
        X["longitude"] = X["longitude"].replace(0, np.nan)

        X["amount_tsh_zero"] = (X["amount_tsh"] == 0).astype(int)
        X["amount_tsh_log"] = np.log1p(X["amount_tsh"])

        X["construction_year"] = X["construction_year"].replace(0, np.nan)

        date = pd.to_datetime(X["date_recorded"], errors="coerce")
        X["recorded_year"] = date.dt.year
        X["recorded_month"] = date.dt.month

        columns_to_drop = [
            "recorded_by", "wpt_name", "subvillage",
            "quantity_group", "source_type", "source_class",
            "extraction_type_group", "extraction_type_class",
            "waterpoint_type_group", "quality_group",
            "payment_type", "management_group",
            "date_recorded"
        ]

        X = X.drop(columns=columns_to_drop)

        categorical_cols = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns

        X[categorical_cols] = (
            X[categorical_cols]
            .fillna("Unknown")
            .astype(str)
        )

        return X


class RareCategoryGrouper(BaseEstimator, TransformerMixin):

    def __init__(self, columns, min_frequency=10):
        self.columns = columns
        self.min_frequency = min_frequency

    def fit(self, X, y=None):
        self.frequent_categories_ = {}

        for col in self.columns:
            counts = X[col].value_counts()

            self.frequent_categories_[col] = (
                counts[counts > self.min_frequency].index
            )

        return self

    def transform(self, X):
        X = X.copy()


        for col in self.columns:
            frequent_categories = self.frequent_categories_[col]

            X[col] = X[col].where(
                X[col].isin(frequent_categories),
                "Other"
            )

        return X