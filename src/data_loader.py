# src/data_loader.py
import pandas as pd
from pathlib import Path
import utils

def load_churn_data(
        filepath : Path | str
) -> pd.DataFrame:
    """
    Loads churn data

    Args
    ---
        filepath : Path | str
            Path of the file from the project root
    
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
            File type is not supported
    """

    path_full : Path = utils.path_from_root(filepath)

    if not path_full.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    df : pd.DataFrame 
    
    ext : str = path_full.suffix
    match ext:
        case '.xlsx':
            df = pd.read_excel(path_full)
        case '.csv':
            df = pd.read_csv(path_full)
        case '.parquet':
            df = pd.read_parquet(path_full)
        case _:
            raise TypeError(f'Invalid file type ({ext})')

    if df.empty:
        raise ValueError(f"Empty dataset")

    return df

def export_churn_data(
        df : pd.DataFrame,
        filepath : str | Path,
        override : bool = True,
        csv_separator : str = ','
) -> None:
    """
    Exports churn data

    Args
    ---
        df : pd.DataFrame
            DataFrame to be exported
        filepath : Path | str
            Path of the file to be saved
        override : bool

    Raises
    ---
        FileExistsError
            File already exists and override = False
        TypeError
            Attempted to save to an unsupported file type
        """
    path_full : Path = utils.path_from_root(filepath)

    if path_full.exists() and not override:
        raise FileExistsError(f'File already exists, use override = True to save over it')
    
    ext : str = path_full.suffix

    match ext:
        case '.xlsx':
            df.to_excel(path_full)
        case '.csv':
            df.to_csv(path_full, sep=csv_separator)
        case '.parquet':
            df.to_parquet(path_full)
        case _:
            raise TypeError(f'Invalid file type ({ext})')

    

def format_column_names(cols : list[str]) -> list[str]:
    """Returns a list with the renamed columns in order"""
    new_cols : list[str] = []
    for col in cols:
        new_col : str = utils.format_column_name(col)
        new_cols += [new_col]

    return new_cols

def preprocess_data(df: pd.DataFrame):
    pass

def get_feature_names(df: pd.DataFrame):
    pass
# def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
#     """
#     Preprocessa dados separando features e target.
    
#     Args:
#         df: DataFrame com dados brutos.
        
#     Returns:
#         Tupla (features DataFrame, target Series).
        
#     Raises:
#         KeyError: Se coluna target não existe.
#     """
#     target_col = "default payment next month"
    
#     if target_col not in df.columns:
#         raise KeyError(f"Coluna target '{target_col}' não encontrada")
    
#     # Remover ID e target das features
#     feature_cols = [c for c in df.columns if c not in ["ID", target_col]]
    
#     X = df[feature_cols]
#     y = df[target_col]
    
#     return X, y


# def get_feature_names(df: pd.DataFrame) -> List[str]:
#     """
#     Retorna lista de nomes de features (exclui ID e target).
    
#     Args:
#         df: DataFrame com dados.
        
#     Returns:
#         Lista de nomes de colunas.
#     """
#     exclude = ["ID", "default payment next month"]
#     return [c for c in df.columns if c not in exclude]
