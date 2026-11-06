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
		string : str,
		lowercase : bool = True
) -> str:
	"""Formats the given string so that it has no special characters"""
	# Source - https://stackoverflow.com/a/23996414
	# Posted by Kobi K, modified by community. See post 'Timeline' for change history
	# Retrieved 2026-11-06, License - CC BY-SA 4.0

	replace_dict : dict  = {
		'ãáà': 'a',
		'óõò': 'o',
		'éèê': 'e',
		'ìí': 'i',
		'ùú': 'u',
		'ç': 'c',
		'/_':' '
		# ' ':'_'
	}
	new_string : str = string
	for key in replace_dict.keys():
		new_string = re.sub(f'[{key}]', replace_dict[key], new_string)
	new_string = re.sub('[^a-zA-Z0-9 \n]', '', new_string).strip()
	if lowercase:
		new_string = new_string.lower()

	return new_string

def remove_double_spaces(
		string : str
) -> str:
	"""Formats the given string so that it has no duplicate spaces"""
	split_string : list[str] = string.split(' ')
	split_string_clean : list[str] = [part for part in split_string if part != '']
	new_string : str = ' '.join(split_string_clean)

	return new_string

def format_column_name(
		string : str,
		lowercase : bool = True
):
	"""Formats the given string so that it has no special characters and no spaces"""

	replace_dict : dict = {
		' ':'_'
	}

	new_string : str = string
	new_string = handle_special_characters(new_string, lowercase)
	new_string = remove_double_spaces(new_string)

	for key in replace_dict.keys():
		new_string = re.sub(f'[{key}]', replace_dict[key], new_string)

	return new_string