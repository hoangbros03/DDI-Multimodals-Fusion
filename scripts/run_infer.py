import logging
import json
from datetime import datetime

import click
import torch

from ddi_kt_2024.preprocess.spacy_nlp import SpacyNLP
from ddi_kt_2024.dependency_parsing.dependency_parser import DependencyParser
from ddi_kt_2024.dependency_parsing.path_processer import PathProcesser
from ddi_kt_2024.model.trainer import Trainer
from ddi_kt_2024.reader.yaml_reader import get_yaml_config
from ddi_kt_2024 import logging_config
from ddi_kt_2024.model.word_embedding import WordEmbedding
from ddi_kt_2024.utils import (
    get_lookup, 
    DictAccessor, 
    standardlize_config,
    get_decode_a_label,
)

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%d%m%y_%H%M")

@click.command()
@click.option("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
def inference(json_path, model_path, result_file, config):
    """
    For inference purpose.
    Json file should has the following content:
    [
        {
            "id",
            "text",
            "e1",
            "e2"
        },
        {} other objects, if any
    ]
    """
    # Load files
    with open(json_path, "r") as f:
        content = json.load(f)
    if not isinstance(content, list) and isinstance(content, dict):
        content = [content]
    logging.info(f"Load json file successfully.")
    config = get_yaml_config(config)
    
    # Initialize
    lookup_word = get_lookup(config['lookup_word'])
    lookup_tag = get_lookup(config['lookup_tag'])
    lookup_dep = get_lookup(config['lookup_dep'])
    lookup_direction = get_lookup(config['lookup_direction'])
    we = WordEmbedding(fasttext_path=config['fasttext_path'],
                   vocab_path=config['vocab_path'])
    dependency_parser = DependencyParser(SpacyNLP().nlp)
    path_processor = PathProcesser(SpacyNLP(), lookup_word, lookup_dep, lookup_tag, lookup_direction)
    logging.info(f"Load dependency parser and path processor successfully.")

    # Load model
    config = DictAccessor(config)
    config = standardlize_config(config)
    model = Trainer(we,
        dropout_rate=config.dropout_rate,
        word_embedding_size=config.word_embedding_size,
        tag_number=config.tag_number,
        tag_embedding_size=config.tag_embedding_size,
        position_number=config.position_number,
        position_embedding_size=config.position_embedding_size,
        direction_number=config.direction_number,
        direction_embedding_size=config.direction_embedding_size,
        edge_number=config.edge_number,
        edge_embedding_size=config.edge_embedding_size,
        token_embedding_size=config.token_embedding_size,
        dep_embedding_size=config.dep_embedding_size,
        conv1_out_channels=config.conv1_out_channels,
        conv2_out_channels=config.conv2_out_channels,
        conv3_out_channels=config.conv3_out_channels,
        conv1_length=config.conv1_length,
        conv2_length=config.conv2_length,
        conv3_length=config.conv3_length,
        w_false=config.w_false,
        w_advice=config.w_advice,
        w_effect=config.w_effect,
        w_mechanism=config.w_mechanism,
        w_int=config.w_int,
        target_class=5,
        lr=config.lr,
        weight_decay=config.weight_decay,
        device=config.device,
        wandb_available=False).model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    logging.info("Model loaded successfully.")
    
    # Inference process
    result_arr = []
    model.eval()
    for sample in content:
        path = dependency_parser.get_sdp_one(sample)
        sample = path_processor.create_mapping(sample, path)

        # Calculate the amount of padding required for each dimension
        sample = torch.nn.functional.pad(sample, (0, 0, 0, 16-sample.shape[0]))
        sample = torch.unsqueeze(sample,0)
        output = model(sample.to(config.device))
        result_arr.append(get_decode_a_label(torch.argmax(output,dim=1)))
    logging.info("Inference process done!")
    with open(result_file, "w") as f:
        json.dump(result_arr, f)
    logging.info(f"Output file has been created at {result_file}")

if __name__=="__main__":
    inference()