"""
Read yaml file
"""
import logging
import yaml

from bc5_2024 import logging_config

def get_yaml_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f.read())

    return dict(config)


def save_yaml_config(file_path, data_content):
    with open(file_path, "w") as f:
        yaml.dump(data_content, f)
    logging.info(f"Yaml file is saved into {file_path}")