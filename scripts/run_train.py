"""
Containing script to train
"""
from pathlib import Path
import logging

import torch
import click
from tqdm import tqdm
from pprint import pprint
from torch.utils.data import DataLoader

from ddi_kt_2024.utils import (
    load_pkl,
    get_labels
)
from ddi_kt_2024.reader.yaml_reader import get_yaml_config
from ddi_kt_2024.model.custom_dataset import CustomDataset, BertEmbeddingDataset
# from ddi_kt_2024.model.trainer import Trainer, BertTrainer
from ddi_kt_2024.model.word_embedding import WordEmbedding
from ddi_kt_2024.multimodal.IFPC import *
from ddi_kt_2024.text.preprocess.asada_preprocess import _negative_filtering
from ddi_kt_2024.multimodal.desc_handler import *
from wandb_setup import wandb_setup
from ddi_kt_2024 import logging_config
from ddi_kt_2024.utils import standardlize_config

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from ddi_kt_2024.text.preprocess.asada_preprocess import *
from ddi_kt_2024.text.reader.yaml_reader import get_yaml_config
from ddi_kt_2024.mol.fn_group_dataset import FnGroupDataset
from ddi_kt_2024.mol.preprocess import mapped_property_reader, get_property_dict, find_drug_property, candidate_property
from ddi_kt_2024.utils import get_labels
from ddi_kt_2024.mol.formula_dataset import FormulaDataloader
from scripts.wandb_setup import wandb_setup
from torch_geometric.loader import DataLoader as GeoDataLoader
from ddi_kt_2024.mol.mol_dataset import MolDataset
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

@click.command()
@click.option("--yaml_path", required=True, type=str, help="Path to the yaml config")
def legacy_run_train(yaml_path):
    # Initialize
    config = get_yaml_config(yaml_path)
    config, wandb_available = wandb_setup(config)
    if not wandb_available:
        config = standardlize_config(config)

    # breakpoint()
    # Load pkl files
    all_candidates_train = load_pkl(config.all_candidates_train)
    all_candidates_test = load_pkl(config.all_candidates_test)
    sdp_train_mapped = load_pkl(config.sdp_train_mapped)
    sdp_test_mapped = load_pkl(config.sdp_test_mapped)
    we = WordEmbedding(fasttext_path=config.fasttext_path,
                   vocab_path=config.vocab_path)

    # Data preparation
    y_train = get_labels(all_candidates_train)
    y_test = get_labels(all_candidates_test)
    if config.type_embed == 'fasttext':
        data_train = CustomDataset(sdp_train_mapped, y_train)
        data_train.fix_exception()
        data_train.batch_padding(batch_size=config.batch_size, min_batch_size=config.min_batch_size)
        data_train.squeeze()
        data_test = CustomDataset(sdp_test_mapped, y_test)
        data_test.fix_exception()
        data_test.batch_padding(batch_size=config.batch_size, min_batch_size=config.min_batch_size)
        data_test.squeeze()
    elif config.type_embed == 'bert_sentence':
        data_train = torch.load(config.train_custom_dataset)
        data_test = torch.load(config.test_custom_dataset)
        # breakpoint()
    else:
        raise ValueError("Value of type_embed isn't supported yet!")
    dataloader_train = DataLoader(data_train, batch_size=config.batch_size)
    dataloader_test = DataLoader(data_test, batch_size=config.batch_size)
    
    # Model initialization
    if config.type_embed == 'fasttext':
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
            wandb_available=wandb_available)
    elif config.type_embed == "bert_sentence":
        model = BertTrainer(
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
            wandb_available=wandb_available)
    model.config = config

    # Model train
    model.train(dataloader_train, dataloader_test, num_epochs=config.epochs)

@click.command()
@click.option("--yaml_path", required=True, type=str, help="Path to the yaml config")
def run_train(yaml_path):
    # Initialize
    config = get_yaml_config(yaml_path)
    config, wandb_available = wandb_setup(config)
    if not wandb_available:
        config = standardlize_config(config)

    # Text handle
    test_candidates = load_pkl(config.candidate_test_path)
    examples = new_convert_to_examples(test_candidates, "test", drugother_mask="full")
    data_text_test = preprocess(examples, model_name=config.model_name_or_path, save_path="data_test.pt")

    train_candidates = load_pkl(config.candidate_train_path)
    examples = new_convert_to_examples(train_candidates, "train", drugother_mask="full")
    data_text_train = preprocess(examples, model_name=config.model_name_or_path, save_path="data_train.pt")
    
    data_text_train = _negative_filtering(data_text_train, data_type="train")
    data_text_test = _negative_filtering(data_text_test, data_type="test")
    
    dataloader_text_train = DataLoader(data_text_train, batch_size=config.batch_size, shuffle=False)
    dataloader_text_test = DataLoader(data_text_test, batch_size=config.batch_size, shuffle=False)

    # Desc handle
    desc_dict = torch.load(config.desc_dict_path)
    data_desc_train_1, data_desc_train_2 = Desc_Fast(train_candidates, desc_dict, 'train', 'e1'), Desc_Fast(train_candidates, desc_dict, 'train', 'e2')
    data_desc_test_1, data_desc_test_2 = Desc_Fast(test_candidates, desc_dict, 'test', 'e1'), Desc_Fast(test_candidates, desc_dict, 'test', 'e2')
    dataloader_train_desc1 = DataLoader(data_desc_train_1, batch_size=config.batch_size, shuffle=False)
    dataloader_train_desc2 = DataLoader(data_desc_train_2, batch_size=config.batch_size, shuffle=False)
    dataloader_test_desc1 = DataLoader(data_desc_test_1, batch_size=config.batch_size, shuffle=False)
    dataloader_test_desc2 = DataLoader(data_desc_test_2, batch_size=config.batch_size, shuffle=False)

    # Image
    data_train = torch.load(config.image_data_train_path)
    data_test = torch.load(config.image_data_test_path)

    data_train.prepare_type = "train"
    data_train.negative_instance_filtering()
    data_test.prepare_type = "test"
    data_test.negative_instance_filtering()
    dataloader_image_train = DataLoader(data_train, batch_size=config.batch_size, shuffle=False)
    dataloader_image_test = DataLoader(data_test, batch_size=config.batch_size, shuffle=False)

    # Formula
    df = mapped_property_reader('cache/mapped_drugs/DDI/ease_matching/full.csv')
    mapped_formula = get_property_dict(df, property_name='formula')
    x_train, y_train = candidate_property(all_candidates_train, mapped_formula)
    x_test, y_test = candidate_property(all_candidates_test, mapped_formula)

    dataloader_train_for1 = FormulaDataloader(x_train,
                                            batch_size=config.batch_size,
                                            nit_path='cache/filtered_ddi/train_filtered_index.txt',
                                            split_option='2',
                                            element=1,
                                            device='cuda')

    dataloader_train_for2 = FormulaDataloader(x_train,
                                            batch_size=config.batch_size,
                                            nit_path='cache/filtered_ddi/train_filtered_index.txt',
                                            split_option='2',
                                            element=2,
                                            device='cuda')

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

    # Graph
    mapped_smiles = get_property_dict(df, property_name='smiles')
    x_train, y_train = candidate_property(all_candidates_train, mapped_smiles)
    x_test, y_test = candidate_property(all_candidates_test, mapped_smiles)

    dataset_train_mol1 = MolDataset(x_train, element=1)
    dataset_train_mol2 = MolDataset(x_train, element=2)
    dataset_test_mol1 = MolDataset(x_test, element=1)
    dataset_test_mol2 = MolDataset(x_test, element=2)

    dataset_train_mol1.negative_instance_filtering('cache/filtered_ddi/train_filtered_index.txt')
    dataset_train_mol2.negative_instance_filtering('cache/filtered_ddi/train_filtered_index.txt')
    dataset_test_mol1.negative_instance_filtering('cache/filtered_ddi/test_filtered_index.txt')
    dataset_test_mol2.negative_instance_filtering('cache/filtered_ddi/test_filtered_index.txt')

    dataloader_train_graph1 = DataLoader(dataset_train_mol1, batch_size=config.batch_size, shuffle=False)
    dataloader_train_graph2 = DataLoader(dataset_train_mol2, batch_size=config.batch_size, shuffle=False)
    dataloader_test_graph1 = DataLoader(dataset_test_mol1, batch_size=config.batch_size, shuffle=False)
    dataloader_test_graph2 = DataLoader(dataset_test_mol2, batch_size=config.batch_size, shuffle=False)

    # Load kwarg
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
            'use_edge_attr': config.gnn_use_edge_attr, # BELOW SETTING MODALS
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
                    freeze_bert=True, # Let it default
                    image_reduced_dim=represent_dim,
                    loss_type=config.loss_type,
                    apply_mask=config.apply_mask,
                    represent_dim = config.represent_dim,
                    fusion_module=config.fusion_module,
                    **kwargs)

    model.config = config

    all_candidates_test = load_pkl(config.candidate_test_path)
    full_labels = get_labels(all_candidates_test)
    with open(config.filtered_index_path, 'r') as f:
        lines = f.read().split('\n')[:-1]
        filtered_lst_index_test = [int(x.strip()) for x in lines]

    gnn_state = torch.load(gnn_state_path,map_location="cpu")
    formula_state = torch.load(formula_state_path,map_location="cpu")
    desc_state = torch.load(desc_state_path,map_location="cpu")
    image_state = torch.load(image_state_path,map_location="cpu")

    if kwargs['freeze_gnn']:
        del gnn_state['classifier.weight']
        del gnn_state['classifier.bias']
        model.model.load_state_dict(gnn_state, strict=False)

    if kwargs['freeze_formula']:
        del formula_state['classifier.weight']
        del formula_state['classifier.bias']
        model.model.load_state_dict(formula_state, strict=False)

    if kwargs['freeze_desc']:
        model.model.load_state_dict(desc_state, strict=False)

    model.train(
        training_loader=dataloader_text_train,
        validation_loader=dataloader_text_test,
        num_epochs=config.epochs,
        train_mol1_loader=dataloader_train_graph1,
        train_mol2_loader=dataloader_train_graph2,
        test_mol1_loader=dataloader_test_graph1,
        test_mol2_loader=dataloader_test_graph2,
        train_mol3_loader=dataloader_train_for1,
        train_mol4_loader=dataloader_train_for2,
        test_mol3_loader=dataloader_test_for1,
        test_mol4_loader=dataloader_test_for2,
        train_mol5_loader=dataloader_train_desc1,
        train_mol6_loader=dataloader_train_desc2,
        test_mol5_loader=dataloader_test_desc1,
        test_mol6_loader=dataloader_test_desc2,
        train_image_loader=dataloader_image_train,
        test_image_loader=dataloader_image_test,
        filtered_lst_index=filtered_lst_index_test,
        full_label=full_labels,
        modal1='graph',
        modal2='formula',
        modal3='desc'
    )
if __name__=="__main__":
    run_train()
