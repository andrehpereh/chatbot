import json
import yaml
import re

def merge_cloudbuild_files(child_files, descriptions, master_filepath="master_cloudbuild.json", timeout_hours = 60 * 60):
    """
    Merges multiple Cloud Build YAML files into a single master file, adding component descriptions.

    Args:
        child_files (list): A list of paths to the child Cloud Build YAML files.
        descriptions (list) : A list of descriptions, corresponding to each child file.
        master_filepath (str, optional): The desired filepath for the output master file. 
                                         Defaults to "master_cloudbuild.yaml".

    Raises:
        FileNotFoundError: If any of the child files cannot be found.
        ValueError: If any of the child files is not valid YAML, or if the number of descriptions
                    doesn't match the number of child files.
    """
    substitutions = []
    variable_pattern = re.compile(r'\$_[A-Z_]+')

    if len(child_files) != len(descriptions):
        raise ValueError("Number of descriptions must match the number of child files")

    master_config = {'steps': []}

    for child_file, description in zip(child_files, descriptions):
        try:
            with open(child_file, 'r') as f:
                child_config = yaml.safe_load(f)

            if 'steps' not in child_config:
                raise ValueError(f"Invalid Cloud Build format in '{child_file}': missing 'steps' key")
            variables = variable_pattern.findall(json.dumps(child_config))
            for variable in variables:
                substitutions.append(variable[1:])
            # Add description as a comment before the steps 
            # master_config['steps'].append({'name': f'# --- {description} ---'})

            # Indentation fix: Directly append the steps from the child config
            master_config['steps'].extend(child_config['steps'])
            

        except FileNotFoundError:
            raise FileNotFoundError(f"Child Cloud Build file not found: '{child_file}'")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in '{child_file}': {e}")
    print("This is all", master_config)
    master_config['timeout'] = f'{timeout_hours * 60 * 60}s'
    with open(master_filepath, 'w') as f:
        json.dump(master_config, f, indent=2)
    # flattened_list = [item for sublist in substitutions for item in sublist]
    return set(substitutions)


def find_missing_elements(my_set, long_string):
    """
    This function finds elements from a set that are not present in a long string.

    Args:
        my_set: A set of strings to check for presence.
        long_string: A string that might contain elements from the set.

    Returns:
        A list of elements from the set that are missing in the long string.
    """
    missing_elements = set(my_set)
    for part in long_string.split(','):
        for element in part.split('='):
            element = element.strip()
            if element in missing_elements:
                missing_elements.remove(element)
    return list(missing_elements)
