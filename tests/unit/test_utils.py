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
        "input_string, expected", [
            ('sããão Paulo', 'saaao_paulo'),
            ('$#**92abc', '92abc'),
            ('não Querías', 'nao_querias')
        ]
    )
    def test_handle_special_characters(self, input_string, expected):
        assert utils.handle_special_characters(input_string) == expected