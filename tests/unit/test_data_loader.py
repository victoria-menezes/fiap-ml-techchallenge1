import pytest
import data_loader

class TestFormatColumnNames:
    @pytest.mark.parametrize(
        "input_cols, expected",[
            (['Nome Completo', 'gênero/ Sexo', 'Localização'], ['nome_completo', 'genero_sexo', 'localizacao'])
        ]
    )
    def test_format_column_names_works(self, input_cols, expected):
        assert data_loader.format_column_names(input_cols) == expected