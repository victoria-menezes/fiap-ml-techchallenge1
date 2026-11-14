import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    OneHotEncoder
    )
from sklearn.compose import (
    ColumnTransformer
)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# custom modules
import data_loader

class ColumnNameFormatter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(data_loader.format_column_names(X.columns), dtype=object)
        return self
    
    def transform(self, X):
        X = X.set_axis(data_loader.format_column_names(X.columns), axis=1)
        missing = set(self.feature_names_in_) - set(X.columns)
        if missing:
            raise ValueError(f'Missing columns present at fit: {sorted(missing)}')
        return X[list(self.feature_names_in_)]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_

class ToFloat(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
            self.feature_names_in_ = np.asarray(data_loader.format_column_names(X.columns), dtype=object)
            return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col, series in X.items():
            if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
                X_copy[col] = series.str.lower()
                X_copy[col] = X_copy[col].replace({'yes':1, 'no':0}).astype(float)
            elif pd.api.types.is_bool_dtype(series) or pd.api.types.is_integer_dtype(series):
                X[col] = series.astype(float)
        return X_copy
        
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_
    
class CollapseIntoNo(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(data_loader.format_column_names(X.columns), dtype=object)
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, series in X.items():
            if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
                X_copy[col] = series.where(
                    series.isin(['Yes', 'yes', 'No', 'no', 1, 0]),
                    other='no'
                )
        return X_copy
        
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_

def get_target_name(processed:bool=False) -> str:
    target : str = 'Churn Value'
    if processed:
        return data_loader.format_column_names([target])[0]
    else:
        return target

def get_feature_names(processed:bool=False) -> list[str]:
    final_cols : list[str] = [
        'Tenure Months',
        'Monthly Charges',
        'Internet Service',
        'Contract',
        'Payment Method',
        'Partner',
        'Dependents',
        'Paperless Billing',
        'Online Security',
        'Online Backup',
        'Device Protection',
        'Tech Support'
        ]
    if processed:
        return data_loader.format_column_names(final_cols)
    return final_cols

def get_preprocessing_pipeline() -> BaseEstimator:
    final_cols = get_feature_names(processed=True)

    keeper = ColumnTransformer(
        transformers=[('keep', 'passthrough', final_cols)],
        remainder='drop',
        verbose_feature_names_out=False
    )

    onehot_cols = [
        'paperless_billing',
        'payment_method',
        'internet_service',
        'contract'
    ]

    yesno_cols = [
        'online_security',
        'online_backup',
        'device_protection',
        'tech_support',
    ]

    column_transformer_initial = ColumnTransformer(
        transformers=[
            ('onehotencoder', OneHotEncoder(
                    sparse_output=False,
                    handle_unknown='ignore',
                ), onehot_cols),
            ('no collapse', CollapseIntoNo(), yesno_cols)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )

    return Pipeline(steps=[
            ('format columns names', ColumnNameFormatter()),
            ('filter columns', keeper),
            ('encode and collapse no', column_transformer_initial),
            ('all to float', ToFloat())
        ]).set_output(transform='pandas')