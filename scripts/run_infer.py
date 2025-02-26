import logging
import json
from datetime import datetime

import click
import torch

# from ddi_kt_2024.preprocess.spacy_nlp import SpacyNLP
# from ddi_kt_2024.dependency_parsing.dependency_parser import DependencyParser
# from ddi_kt_2024.dependency_parsing.path_processer import PathProcesser
# from ddi_kt_2024.model.trainer import Trainer
from ddi_kt_2024.text.preprocess.asada_preprocess import new_convert_to_examples, preprocess
from ddi_kt_2024.multimodal.IFPC import *
from ddi_kt_2024.text.reader.yaml_reader import get_yaml_config
from ddi_kt_2024 import logging_config
# from ddi_kt_2024.model.word_embedding import WordEmbedding
from ddi_kt_2024.utils import (
    get_lookup, 
    DictAccessor, 
    standardlize_config,
    get_decode_a_label,
)
from ddi_kt_2024.text.preprocess.asada_preprocess import _negative_filtering
from ddi_kt_2024.multimodal.desc_handler import *
from ddi_kt_2024.utils import load_pkl, get_labels
from ddi_kt_2024.mol.fn_group_dataset import FnGroupDataset
from ddi_kt_2024.mol.preprocess import mapped_property_reader, get_property_dict, find_drug_property, candidate_property
from ddi_kt_2024.mol.formula_dataset import FormulaDataloader
from torch_geometric.loader import DataLoader as GeoDataLoader
from ddi_kt_2024.mol.mol_dataset import MolDataset
from torch.utils.data import DataLoader

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%d%m%y_%H%M")

@click.command()
@click.option("--json_path", required=True, type=str, help="Path to the json file")
@click.option("--model_path", required=True, type=str, help="Path to the model file")
@click.option("--result_file", required=True, type=str, help="Path to the result file")
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
    config = DictAccessor(config)
    config = standardlize_config(config)
    kwargs = {
        'seed': config.seed,
        'desc_conv_output_size': config.desc_conv_output_size,
        'desc_conv_window_size': config.desc_conv_window_size,
        'desc_layer_hidden': config.desc_layer_hidden,
        'formula_conv_output_size': config.formula_conv_output_size,
        'formula_conv_window_size': config.formula_conv_window_size,
        'gnn_option': config.gnn_option,
        'readout_option': config.gnn_readout_option,
        'num_layers_gnn': config.gnn_num_layers,
        'hidden_channels': config.gnn_hidden_channels,
        'use_gat_v2': config.gnn_use_gat_v2,
        'use_edge_attr': config.gnn_use_edge_attr,
        'use_graph': config.use_graph,
        'use_formula': config.use_formula,
        'use_desc': config.use_desc,
        'use_image': config.use_image,
        'freeze_gnn': config.freeze_gnn,
        'freeze_formula': config.freeze_formula,
        'freeze_image': config.freeze_image,
        'freeze_desc': config.freeze_desc,
    }
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
                    freeze_bert=True,
                    image_reduced_dim=config.represent_dim[-1],
                    loss_type=config.loss_type,
                    apply_mask=config.apply_mask,
                    represent_dim=config.represent_dim,
                    fusion_module=config.fusion_module,
                    **kwargs)

    model.config = config

    checkpoint = torch.load(model_path)
    model.model.load_state_dict(checkpoint)
    logging.info("Model loaded successfully.")
    
    # Load Modalities
    desc_dict = torch.load(config.desc_dict_path, weights_only=False)
    df = mapped_property_reader('cache/mapped_drugs/DDI/ease_matching/full.csv')
    mapped_formula = get_property_dict(df, property_name='formula')
    mapped_smiles = get_property_dict(df, property_name='smiles')

    # Prepare Data
    result_arr = []
    model.model.eval()
    for sample in content:
        drug_name = sample['e1']  # Assuming 'e1' contains the drug name

        # Check if the drug has associated modalities
        if drug_name in mapped_formula and drug_name in mapped_smiles:
            # Prepare text data
            examples = new_convert_to_examples([sample], "test", drugother_mask="full")
            data_text_test = preprocess(examples, model_name=config.model_name_or_path, save_path=None)
            data_text_test = _negative_filtering(data_text_test, data_type="test")
            dataloader_text_test = DataLoader(data_text_test, batch_size=config.batch_size, shuffle=False)

            # Prepare description data
            data_desc_test_1, data_desc_test_2 = Desc_Fast([sample], desc_dict, 'test', 'e1'), Desc_Fast([sample], desc_dict, 'test', 'e2')
            dataloader_test_desc1 = DataLoader(data_desc_test_1, batch_size=config.batch_size, shuffle=False)
            dataloader_test_desc2 = DataLoader(data_desc_test_2, batch_size=config.batch_size, shuffle=False)

            # Prepare image data
            data_test = torch.load(config.image_data_test_path, weights_only=False)
            data_test.prepare_type = "test"
            data_test.negative_instance_filtering()
            dataloader_image_test = DataLoader(data_test, batch_size=config.batch_size, shuffle=False)

            # Prepare formula data
            x_test, y_test = candidate_property([sample], mapped_formula)
            dataloader_test_for1 = FormulaDataloader(x_test,
                                                    batch_size=config.batch_size,
                                                    nit_path='cache/filtered_ddi/test_filtered_index.txt',
                                                    split_option='2',
                                                    element=1,
                                                    device='cuda')

            dataloader_test_for2 = FormulaDataloader(x_test,
                                                    batch_size=config.batch_size,
                                                    nit_path='cache/filtered_ddi/test_filtered_index.txt',
                                                    split_option='2',
                                                    element=2,
                                                    device='cuda')

            # Prepare graph data
            x_test, y_test = candidate_property([sample], mapped_smiles)
            dataset_test_mol1 = MolDataset(x_test, element=1)
            dataset_test_mol2 = MolDataset(x_test, element=2)
            dataset_test_mol1.negative_instance_filtering('cache/filtered_ddi/test_filtered_index.txt')
            dataset_test_mol2.negative_instance_filtering('cache/filtered_ddi/test_filtered_index.txt')
            dataloader_test_graph1 = GeoDataLoader(dataset_test_mol1, batch_size=config.batch_size, shuffle=False)
            dataloader_test_graph2 = GeoDataLoader(dataset_test_mol2, batch_size=config.batch_size, shuffle=False)

            # Perform inference
            for batch in dataloader_text_test:
                inputs = {
                    'input_ids': batch[0].to(config.device),
                    'attention_mask': batch[1].to(config.device),
                    'token_type_ids': batch[2].to(config.device),
                    'relative_dist1': batch[3].to(config.device),
                    'relative_dist2': batch[4].to(config.device),
                    'labels': batch[5].to(config.device),
                    'graph1': next(iter(dataloader_test_graph1)).to(config.device),
                    'graph2': next(iter(dataloader_test_graph2)).to(config.device),
                    'formula1': next(iter(dataloader_test_for1)).to(config.device),
                    'formula2': next(iter(dataloader_test_for2)).to(config.device),
                    'desc1': next(iter(dataloader_test_desc1)).to(config.device),
                    'desc2': next(iter(dataloader_test_desc2)).to(config.device),
                    'image': next(iter(dataloader_image_test)).to(config.device)
                }
                with torch.no_grad():
                    output = model.model(**inputs)
                    result_arr.append(get_decode_a_label(torch.argmax(output, dim=1)))

    logging.info("Inference process done!")
    with open(result_file, "w") as f:
        json.dump(result_arr, f)
    logging.info(f"Output file has been created at {result_file}")

if __name__ == "__main__":
    inference()