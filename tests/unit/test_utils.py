import pytest
import utils
from pathlib import Path

class TestGetProjectRoot:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Runs automatically before each test method in the class"""
        self.result = utils.get_project_root()

    def test_get_project_root_returns_path(self):
        """Verify the function returns a pathlib.Path instance"""
        assert isinstance(self.result, Path) 

    def test_get_project_root_is_absolute(self):
        """Verify the function returns an absolute Path"""
        assert self.result.is_absolute()

    def test_get_project_root_exists(self):
        """Verify the function returns a path that exists"""
        assert self.result.exists()    

class TestPathFromRoot:
    @pytest.fixture(autouse=True)
    def setup_mock_root(self, monkeypatch, tmp_path):
        """
        Runs automatically before each test method in the class
        Mocks get_project_root() to return a clean temporary directory
        """

        monkeypatch.setattr(utils, "get_project_root", lambda: tmp_path)
        self.tmp_path = tmp_path
        self.result = utils.path_from_root('data')
        self.expected : Path = self.tmp_path / 'data'
    
    def test_path_from_root_returns_path(self):
        """Verify the function returns a pathlib.Path instance"""
        assert isinstance(self.result, Path) 

    def test_path_from_root_is_absolute(self):
        """Verify the function returns an absolute Path"""
        assert self.result.is_absolute()

    def test_path_from_root_correctly_joined(self):
        """Verify the function returns a valid path"""
        assert self.result == self.expected

class TestHandleSpecialCharacters:
    @pytest.mark.parametrize(
        "input_string, input_lowercase, expected", [
            ('sããão Paulo', True, 'saaao paulo'),
            ('$#**92abc', True, '92abc'),
            ('não Querías', False, 'nao Querias')
        ]
    )
    def test_handle_special_characters(self, input_string, input_lowercase, expected):
        assert utils.handle_special_characters(input_string, input_lowercase) == expected

class TestFormatColumnName:
    @pytest.mark.parametrize(
        "input_string, input_lowercase, expected", [
            ('Pastel de Belém', True, 'pastel_de_belem'),
            ('Gênero / Sexo', True, 'genero_sexo'),
            ('não Querías', False, 'nao_Querias')
        ]
    )
    def test_format_column_name(self, input_string, input_lowercase, expected):
        assert utils.format_column_name(input_string, input_lowercase) == expected

class TestRemoveDoubleSpaces:
    @pytest.mark.parametrize(
        "input_string, expected",[
            ('a  b c', 'a b c'),
            ('abc  abc   abc', 'abc abc abc'),
            ('a                      ', 'a')
        ]
    )
    def test_remove_double_spaces(self, input_string, expected):
        assert utils.remove_double_spaces(input_string) == expected