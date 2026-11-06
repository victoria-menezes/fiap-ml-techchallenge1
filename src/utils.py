from pathlib import Path
import re

##
def get_project_root() -> Path:
    project_root : Path = Path(__file__).resolve().parent.parent

    return project_root

def path_from_root(
        relative_path : Path | str
) -> Path:
    return get_project_root() / relative_path

def handle_special_characters(
        string : str
) -> str:
    # Source - https://stackoverflow.com/a/23996414
    # Posted by Kobi K, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-11-06, License - CC BY-SA 4.0

    replace_dict : dict  = {
        'ãáà': 'a',
        'óõò': 'o',
        'éè': 'e',
        'ìí': 'i',
        'ùú': 'u',
        'ç': 'c',
        ' ':'_'
    }
    new_string : str = string
    for key in replace_dict.keys():
        new_string = re.sub(f'[{key}]', replace_dict[key], new_string)
    new_string = re.sub('[^a-zA-Z0-9_\n]', '', new_string).lower().strip()

    return new_string

def format_column_name (
        col_name : str        
) -> str:
    new_col_name : str = col_name.lower()
    new_col_name = handle_special_characters(new_col_name).strip() 
    return new_col_name