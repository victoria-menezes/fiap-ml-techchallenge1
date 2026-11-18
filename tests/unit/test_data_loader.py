import pytest
from pathlib import Path
import pandas as pd

from data_loader import (
    load_churn_data,
    export_churn_data,
    format_column_names
    )
import utils

class TestLoadChurnData:
    @pytest.fixture(autouse = True)
    def setup(self, tmp_path):
        """Runs before each test method"""
        self.dummy_df : pd.DataFrame = pd.DataFrame(
            {
                'customer_id': [1, 3, 432, 2],
                'category':['A', 'B', None, 'B'],
                'churned':[1, 0, 1, 1]
            }
        )
        self.tmp_path : Path = tmp_path

    @pytest.mark.parametrize(
            "file_name", [
                'dummy.xlsx',
                'abc def.csv',
                'qwerty.parquet'
            ]
    )
    def test_success(self, monkeypatch, file_name):
        """Verifies a valid file with data returns non-empty DataFrame"""

        # dummy file
        filepath : Path = self.tmp_path / file_name
        ext : str = filepath.suffix
        match ext:
            case '.csv':
                self.dummy_df.to_csv(filepath)
            case '.xlsx':
                self.dummy_df.to_excel(filepath)
            case '.parquet':
                self.dummy_df.to_parquet(filepath)

        # mock path_from_root
        monkeypatch.setattr(utils, "path_from_root", lambda p:filepath)

        # call and assert
        result = load_churn_data(filepath)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) == 4

class TestExportChurnData:
    @pytest.fixture(autouse = True)
    def setup(self, tmp_path):
        """Runs before each test method"""
        self.dummy_df : pd.DataFrame = pd.DataFrame(
            {
                'customer_id': [1, 3, 432, 2],
                'category':['A', 'B', None, 'B'],
                'churned':[1, 0, 1, 1]
            }
        )
        self.tmp_path : Path = tmp_path

    @pytest.mark.parametrize(
        "file_name", [
            'dummy.xlsx',
            'abc def.csv',
            'qwerty.parquet'
        ]
    )
    def test_success(self, monkeypatch, file_name):
        """Verifies a valid path outputs successfully"""
        # dummy file
        filepath : Path = self.tmp_path / file_name
        ext : str = filepath.suffix

        # mock path_from_root
        monkeypatch.setattr(utils, "path_from_root", lambda p:filepath)
        export_churn_data(
            df = self.dummy_df,
            filepath = filepath
        )

        exported : pd.DataFrame

        match ext:
            case '.csv':
                exported = pd.read_csv(filepath)
            case '.xlsx':
                exported = pd.read_excel(filepath)
            case '.parquet':
                exported = pd.read_parquet(filepath)
        assert isinstance(exported, pd.DataFrame)
        assert not exported.empty
        assert len(exported) == 4

class TestFormatColumnNames:
    @pytest.mark.parametrize(
        "input_cols, expected",[
            (['Nome Completo', 'gênero/ Sexo', 'Localização'], ['nome_completo', 'genero_sexo', 'localizacao'])
        ]
    )
    def test_format_column_names_success(self, input_cols, expected):
        """Verifies the function outputs expected results"""
        assert format_column_names(input_cols) == expected

