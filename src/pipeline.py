"""Preprocessing and model pipelines for the churn dataset.
 
Pipeline steps (names are stable, so grid-search keys such as
``preprocessing__scale columns__...`` keep working):
 
1. format columns names  - rename columns with data_loader.format_column_names
2. filter columns        - keep only the model features
3. encode and collapse no - one-hot the categoricals, collapse Yes/No-like
                            columns ('No internet service' -> 'no')
4. all to float          - Yes/No strings and ints/bools -> float
5. scale columns         - (optional) standardize the numeric columns
 
Every step is fit inside the pipeline, so cross-validating the whole pipeline
does not leak information from validation folds.
"""

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
from sklearn.preprocessing import StandardScaler

# custom modules
from data_loader import format_column_names

_TARGET: str = 'Churn Value'

_FEATURES: list[str] = [
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

_ONEHOT: list[str] = [
    'Payment Method',
    'Internet Service',
    'Contract'
]

_YESNO: list[str] = [
    'Online Security',
    'Online Backup',
    'Device Protection',
    'Tech Support',
    'Paperless Billing',
    'Partner',
    'Dependents'
]

_SCALE: list[str] = [
    'Tenure Months',
    'Monthly Charges'
]

_unknown = set(_ONEHOT + _YESNO + _SCALE) - set (_FEATURES)

if _unknown:
    raise ValueError(f'Column groups reference columns not in _FEATURES {sorted(_unknown)}')

_KEPT_VALUES: tuple[str, ...] = ('yes', 'no')
_TO_FLOAT: dict[str, float] = {
    'yes': 1.0,
    'no': 0.0,
    '1': 1.0,
    '0': 0.0
}

### Utils
def _format(column_names) -> list[str]:
    """Formats raw column names with data_loader.format_column_names, as a list"""
    return list(format_column_names(column_names))

def _is_text(series: pd.Series) -> bool:
    """Checks if series is of string/object type"""
    return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)

def _require_dataframe(X) -> None:
    """Raises TypeError if X is not a pandas DataFrame"""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f'Expected pandas DataFrame, got {type(X).__name__}')

### Transformers
class ColumnNameFormatter(BaseEstimator, TransformerMixin):
    """Renames columns with data_loader.format_column_names and fixes their order.
 
    ``fit`` memorizes the formatted column names of the training data.
    ``transform`` formats the incoming names and returns exactly those columns,
    in the same order. Extra columns are dropped.
 
    Attributes
    ---
        feature_names_out_ : np.ndarray
            Formatted column names seen during fit
    """
    def fit(self, X, y=None):
        """Memorizes the formatted column names.
 
        Raises
        ---
            TypeError
                If X is not a DataFrame
            ValueError
                If formatting makes two column names collide
        """
        _require_dataframe(X)
        names = _format(X.columns)
        if len(set(names)) != len(names):
            raise ValueError('Formatted column names are not unique')
        self.feature_names_out_ = np.asarray(names, dtype=object)
        return self
    
    def transform(self, X):
        """Returns X with formatted names, restricted to the columns seen in fit.
 
        Raises
        ---
            TypeError
                If X is not a DataFrame
            ValueError
                If a column present at fit is missing
        """
        _require_dataframe(X)
        X = X.set_axis(_format(X.columns), axis=1)
        missing = set(self.feature_names_out_) - set(X.columns)
        if missing:
            raise ValueError(f'Missing columns present at fit: {sorted(missing)}')
        return X[list(self.feature_names_out_)]

    def get_feature_names_out(self, input_features=None):
        """Returns the formatted column names."""
        return self.feature_names_out_

class _KeepsColumns(BaseEstimator, TransformerMixin):
    """Base for stateless transformers that return the same columns they receive.
 
    Attributes
    ---
        feature_names_in_ : np.ndarray
            Column names seen during fit (also the output names)
    """
 
    def fit(self, X, y=None):
        """Memorizes the column names. Raises TypeError if X is not a DataFrame."""
        _require_dataframe(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self
 
    def get_feature_names_out(self, input_features=None):
        """Returns the column names, unchanged from the input."""
        return self.feature_names_in_

class ToFloat(BaseEstimator, TransformerMixin):
    """Converts every column to float and formats the column names.
 
    * string/object columns: 'yes'/'no' (any case, surrounding spaces ignored)
      and '1'/'0' become 1.0/0.0. Missing values stay NaN.
    * bool, integer and float columns: cast with ``astype(float)``.
    * column names go through data_loader.format_column_names, which turns the
      one-hot names (e.g. 'payment_method_Credit card') into clean snake_case.
 
    Attributes
    ---
        feature_names_out_ : np.ndarray
            Formatted column names seen during fit
    """
    def fit(self, X, y=None):            
        """Memorizes the formatted column names. Raises TypeError if X is not a DataFrame."""
        _require_dataframe(X)

        self.feature_names_out_ = np.asarray(format_column_names(X.columns), dtype=object)
        return self
    
    def transform(self, X):
        """Returns a float copy of X.
 
        Raises
        ---
            ValueError
                If a text column has a value other than yes/no/1/0
            TypeError
                If X is not a DataFrame
        """
        _require_dataframe(X)
        X_copy = X.copy()

        for col in X_copy.columns:
            series = X_copy[col]
            if _is_text(series):
                text = series.astype(str).str.strip().str.lower().where(
                        series.notna()
                    )
                mapped = text.map(_TO_FLOAT)
                unmapped = text.notna() & mapped.isna()
                if unmapped.any():
                    bad = sorted(text[unmapped].unique())
                    raise ValueError(
                        f'Column {col!r} has values ToFloat cannot convert: {bad}'
                    )
                X_copy[col] = mapped.astype(float)
            else:
                X_copy[col] = series.astype(float)
        return X_copy.set_axis(_format(X_copy.columns), axis=1)
        
    def get_feature_names_out(self, input_features=None):
        """Returns the formatted column names"""
        return self.feature_names_out_
    
class CollapseIntoNo(_KeepsColumns):
    """Normalizes Yes/No-like text columns, mapping everything that is not a yes to 'no'.
 
    Text values are lower-cased and stripped. 'yes', 'no', '1' and '0' are kept;
    any other value ('No internet service', 'No phone service', ...) and any
    missing value becomes 'no'. Non-text columns are left untouched.
    """

    def transform(self, X):        
        """Returns a copy of X with text columns collapsed.
 
        Raises
        ---
            TypeError
                If X is not a DataFrame
        """
        _require_dataframe(X)
        X_copy = X.copy()
        for col in X_copy.columns:
            series = X_copy[col]
            if _is_text(series):
                text = series.astype(str).str.strip().str.lower()
                X_copy[col] = text.where(text.isin(_KEPT_VALUES), 'no')
        return X_copy

### Public

def get_target_name(processed:bool=False) -> str:
    """Returns the target column name.
 
    Parameters
    ---
        processed : bool, default False
            If True, returns the name after data_loader.format_column_names
 
    Returns
    ---
        str
    """
    return _format([_TARGET])[0] if processed else _TARGET

def get_feature_names(processed:bool=False) -> list[str]:
    """Returns the model's feature column names, in order.
 
    Parameters
    ---
        processed : bool, default False
            If True, returns the names after data_loader.format_column_names
 
    Returns
    ---
        list[str]
            A new list on every call, safe to modify
    """
    return _format(_FEATURES) if processed else list(_FEATURES)

def get_preprocessing_pipeline(scale : bool = False) -> BaseEstimator:
    """Builds the preprocessing pipeline (unfitted). Output is a pandas DataFrame.
 
    Expects a DataFrame whose columns, once formatted, include every feature
    in get_feature_names().
 
    Parameters
    ---
        scale : bool, default False
            If True, standardizes the numeric columns (tenure and monthly
            charges). Needed for scale-sensitive models such as logistic
            regression; unnecessary for tree-based models
 
    Returns
    ---
        Pipeline
    """
    keep_features = ColumnTransformer(
        [('keep', 'passthrough', get_feature_names(processed=True))],
        remainder='drop',
        verbose_feature_names_out=False
    )

    onehot_and_collapse = ColumnTransformer(
        [
            (
                'onehotencoder',
                OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
                _format(_ONEHOT)
            ),
            (
                ('no collapse', CollapseIntoNo(), _format(_YESNO))
            )
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )

    steps : list = [
        ('format column names', ColumnNameFormatter()),
        ('filter columns', keep_features),
        ('encode and collapse no', onehot_and_collapse),
        ('all to float', ToFloat())
    ]

    if scale:
        steps.append((
            'scale columns', ColumnTransformer(
                    [(('standard scaler'), StandardScaler(), _format(_SCALE))],
                    remainder='passthrough',
                    verbose_feature_names_out=False
                )
            ))

    return Pipeline(steps=steps).set_output(transform='pandas')

def build_model_pipeline(estimator, scale:bool=False) -> Pipeline:
    """Chains the preprocessing pipeline and an estimator into one Pipeline.
 
    Cross-validating or grid-searching this pipeline refits the preprocessing
    on each training fold, so no validation data leaks into it.
 
    Parameters
    ---
        estimator : estimator
            Any scikit-learn classifier
        scale : bool, default False
            Forwarded to get_preprocessing_pipeline
 
    Returns
    ---
        Pipeline
            Steps 'preprocessing' and 'model'
    """
    return Pipeline([
            ('preprocessing', get_preprocessing_pipeline(scale=scale)),
            ('model', estimator)
        ])