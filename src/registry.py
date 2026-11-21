"""Module dedicated to saving the model and relevant statistics (such as threshold)"""

import json
import platform
from datetime import date

import joblib
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score

from utils import create_path_from_root, path_from_root


def _json_default(
        obj : np.integer | np.floating | np.ndarray
    ):
    """Converts obj, a numpy object, to an int, float or list so it can be
    processed by the json library"""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item() # returns as an int or float
    if isinstance(obj, np.ndarray):
        return obj.tolist() # returns as a list
    raise TypeError(f'Not serializable: {type(obj)}')

def save_champion(
        name : str,
        grid_results : dict,
        y_test,
        feature_names : list[str],
        seed : int,
        n_train : int,
        target_recall : float,
        out_dir = 'models',
        out_name = 'champion'
    ):
    """Saves the model and the model's metadata.
    
    Parameters
    ---
        name : str
            The identifier of the model in grid_results
        grid_results : dict
            Typically a grid_search_models or a random_grid_search_models output.
            Must have been run by add_stats_to_grid_results
        y_test
            The test data true values
        feature_names : list[str]
            The dataset features passed to the Pipeline
        seed : int
        n_train
            The size of the train set
        target_recall : float, default 0.80
        out_dir : str
            The name of the folder to output the results
        out_name : str
            How to name the resulting files.
            Model is saved as {out_name}.joblib,
            and metadata is saved as {out_name}_metadata.json
    """
    path_out = create_path_from_root(out_dir)
    file_name = f'{out_name}.joblib'
    metadata_name = f'{out_name}_metadata.json'

    results = grid_results[name]
    grid_search, stats = results['grid_search'], results['stats']

    joblib.dump(grid_search.best_estimator_, path_out / file_name)

    metadata : dict = {
        'model_name':name,
        'model_file':file_name,
        'created_at':date.today().isoformat(),
        'seed':seed,
        'versions':{
                'python':platform.python_version(),
                'scikit-learn':sklearn.__version__
            },
        'dataset': {
                'n_train':n_train,
                'n_test':len(y_test),
                'churn_rate_test':float(np.mean(y_test))
            },
        'input_features':feature_names,
        'model_features': _json_default(grid_search.best_estimator_[:-1].get_feature_names_out()),
        'threshold':stats['threshold'],
        'threshold_rule': f'highest threshold with recall >= {target_recall} on out-of-fold train predictions',
        'metrics': {
                'cv_average_precision_mean': stats['cv_ap_mean'],
                'cv_average_precision_std': stats['cv_ap_std'],
                'oof_recall': stats['recall'],
                'oof_precision': stats['precision'],
                'test_recall':stats['recall_on_test'],
                'test_precision': stats['precision_on_test'],
                'test_average_precision': average_precision_score(y_test, results['y_proba'])
            },
        'best_params': grid_search.best_params_
    }

    (path_out / metadata_name).write_text(
        json.dumps(metadata, indent = 2, default = _json_default), encoding = 'utf-8'
    )
