import pytest
import data_loader


class TestLoadChurnData:
    pass

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

