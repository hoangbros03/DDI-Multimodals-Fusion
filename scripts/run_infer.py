import logging
import json
from datetime import datetime

import click
import torch

from ddi_kt_2024.preprocess.spacy_nlp import SpacyNLP
from ddi_kt_2024.dependency_parsing.dependency_parser import DependencyParser
from ddi_kt_2024.dependency_parsing.path_processer import PathProcesser
# from ddi_kt_2024.model.trainer import Trainer
from ddi_kt_2024.multimodal.IFPC import *
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
    # Load model
    config = DictAccessor(config)
    config = standardlize_config(config)
    model = Trainer(num_labels=config.target_class,
                    dropout_rate=config.dropout_rate_other,
                    dropout_prob_text=config.dropout_rate_bert_output,
                    out_conv_list_dim=config.out_conv_list_dim,
                    conv_window_size=config.conv_window_size,
                    max_seq_length=config.max_seq_length,
                    pos_emb_dim=config.pos_emb_dim,
                    middle_layer_size=config.middle_layer_size,
                    activation=config.activation,
                    optimizer=config.optimizer,
                    weight_decay=config.weight_decay,
                    model_name_or_path=config.model_name_or_path,
                    wandb_available=True,
                    freeze_bert=True, # Let it default
                    image_reduced_dim=represent_dim,
                    loss_type=config.loss_type,
                    apply_mask=config.apply_mask,
                    represent_dim = config.represent_dim,
                    fusion_module=config.fusion_module,
                    **kwargs)

    model.config = config

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