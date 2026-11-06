# src/data_loader.py
import pandas as pd
from pathlib import Path
from utils import path_from_root

def load_churn_data(
        filepath : Path | str
) -> pd.DataFrame:
    """
    Loads churn data

    Args
    ---
        filepath : Path | str
            Path of an excel file from the project root
    
    Returns
    ---
        pd.Dataframe
            Data in form of a pandas DataFrame

    Raises
    ---
        FileNotFoundError
            File does not exist
        ValueError
            Data is blank
        TypeError
            File is not a .xlsx
    """

    path_full : Path = path_from_root(filepath)

    if not path_full.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if path_full.suffix != '.xlsx':
        raise TypeError(f'File is not a .xlsx file')

    df : pd.DataFrame = pd.read_excel(path_full)

    if df.empty:
        raise ValueError(f"Empty dataset")

    return df

def preprocess_data(df: pd.DataFrame):
    pass

def get_feature_names(df: pd.DataFrame):
    pass
