import pytest
from pathlib import Path
import pandas as pd

import data_loader
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
    def test_load_churn_data_success(self, monkeypatch, file_name):
        """Verifies a valid .xlsx with data returns non-empty DataFrame"""

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
        result = data_loader.load_churn_data(filepath)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert len(result) == 4

class TestExportChurnData:
    pass

class TestFormatColumnNames:
    @pytest.mark.parametrize(
        "input_cols, expected",[
            (['Nome Completo', 'gênero/ Sexo', 'Localização'], ['nome_completo', 'genero_sexo', 'localizacao'])
        ]
    )
    def test_format_column_names_success(self, input_cols, expected):
        """Verifies the function outputs expected results"""
        assert data_loader.format_column_names(input_cols) == expected

