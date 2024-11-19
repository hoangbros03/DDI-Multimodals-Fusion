import pickle as pkl
from pathlib import Path
import logging

import torch
import numpy as np

import bc5_2024.logging_config
from bc5_2024.text.reader.yaml_reader import save_yaml_config

class DictAccessor:
    def __init__(self, data):
        self.data = data

    def __getattr__(self, attr):
        return self.data.get(attr)

    def __getitem__(self, item):
        parts = item.split('.')
        value = self.data
        for part in parts:
            value = value.get(part)
            if value is None:
                return None
        return value


def get_trimmed_w2v_vectors(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        matrix of embeddings (np array)
    """
    with np.load(filename) as data:
        return data['embeddings']
    
def load_vocab(filename):
    """
    Args:
        filename: file with a word per line
    Returns:
        d: dict[word] = index
    """
    d = dict()
    with open(filename) as f:
        for idx, word in enumerate(f):
            word = word.strip()
            d[word] = idx + 1  # preserve idx 0 for pad_tok
    return d


def id_find(lst, id):
    for element in lst:
        if element['@id'] == id:
            return element

def offset_to_idx(text, offset, nlp):
    '''
    Given offset of token in text, return its index in text.
    '''
    doc = nlp(text)
    offset = offset.split(';')[0]
    start = int(offset.split('-')[0])
    end = int(offset.split('-')[1])
    start_idx = -1
    end_idx = -1
    for i in range(len(doc) - 1):
        if doc[i].idx <= start and doc[i+1].idx > start:
            start_idx = doc[i].i
        if doc[i].idx < end and doc[i+1].idx > end:
            end_idx = doc[i].i
    if start_idx == -1:
        start_idx = len(doc) - 1
        end_idx = len(doc) - 1
    assert start_idx != -1, end_idx != -1
    return start_idx, end_idx

def idx_to_offset(text, idx, nlp):
    doc = nlp(text)
    return (doc[idx].idx, doc[idx].idx + len(doc[idx].text))

def get_labels(all_candidates):
    label_list = list()
    for candidate in all_candidates:
        if candidate['label'] == 0:
            label_list.append(torch.tensor([0]))
        elif candidate['label'] == 1:
            label_list.append(torch.tensor([1]))
    return label_list

def get_decode_a_label(result):
    if int(result[0])==0:
        return 'false'
    elif int(result[0])==1:
        return 'advise'
    elif int(result[0])==2:
        return 'effect'
    elif int(result[0])==3:
        return 'mechanism'
    elif int(result[0])==4:
        return 'int'
        
def get_lookup(path):
    '''
    Get lookup table from file
    '''
    with open(path, 'r') as file:
        f = file.read().split('\n')
    return {f[i]: i + 1 for i in range(len(f))}

def lookup(element, dct):
    '''
    Get index of element in lookup table
    '''
    try: 
        idx = dct[element]
    except:
        idx = 0
    return idx

def rm_no_smiles(x, y):
    """
    Remove samples with no smiles mapping in mol dataset
    """
    x_new, y_new = list(), list()
    idx_list = list()
    for i in range(len(x)):
        if x[i][0] != 'None' and x[i][1] != 'None': 
            x_new.append(x[i])
            y_new.append(y[i])
            idx_list.append(i)
            
    return x_new, y_new, idx_list

def load_pkl(path):
    with open(path, 'rb') as file:
        return pkl.load(file)
    
def dump_pkl(obj, path):
    with open(path, 'wb') as file:
        pkl.dump(obj, file)

def standardlize_config(config):
    # Temporary def 
    if isinstance(config.w_false, str):
        config.w_false = eval(config.w_false)
    if isinstance(config.w_advice, str):
        config.w_advice = eval(config.w_advice)
    if isinstance(config.w_effect, str):
        config.w_effect = eval(config.w_effect)
    if isinstance(config.w_mechanism, str):
        config.w_mechanism = eval(config.w_mechanism)
    if isinstance(config.w_int, str):
        config.w_int = eval(config.w_int)
    return config

def check_and_create_folder(path, folder_name=None):
    if folder_name is not None:
        p = Path(Path(path) / folder_name)
    else:
        p = Path(path)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
        logging.info(f"Path {str(p)} has been created!")
    else:
        logging.info(f"Path {str(p)} is already existed!")

def save_model(output_path, file_name, config, model, wandb_available=False):
    """
    The folder structure is following:
    <save_folder>
        config.yaml
        <Save file 1>
        <Save file 2>
    """
    if not Path(output_path).exists():
        check_and_create_folder(output_path)
    # Check if .yaml is existing
    if len(list(Path(output_path).glob("*.yaml"))) ==0:
        # Saving yaml
        if not wandb_available:
            save_yaml_config(str(Path(output_path) / "config.yaml"), config.data)
        else:
            save_yaml_config(str(Path(output_path) / "config.yaml"), config)
    # Save .pt file
    torch.save(model.state_dict(), str(Path(output_path) / file_name))
    logging.info(f"Model saved into {str(Path(output_path) / file_name)}")

def get_idx(sent, vocab_lookup, tag_lookup, direction_lookup, edge_lookup):
    '''
    Get index of features of tokens in sentence
    '''
    if sent == None:
        return None
    to_return = list()
    i = 0
    for dp in sent:
        word1_idx = lookup(dp[0][0], vocab_lookup)
        word2_idx = lookup(dp[2][0], vocab_lookup)
        tag1_idx = lookup(dp[0][1], tag_lookup)
        tag2_idx = lookup(dp[2][1], tag_lookup)
        direction_idx = lookup(dp[1][0], direction_lookup)
        edge_idx = lookup(dp[1][1], edge_lookup)
        pos1 = dp[0][2]
        pos2 = dp[2][2]
        v = torch.tensor([word1_idx, tag1_idx, direction_idx, edge_idx, word2_idx, tag2_idx])
        v = torch.hstack((v[:2], pos1, v[2:6], pos2))
        if i == 0:
            to_return = v.view(1, -1)
        else:
            to_return = torch.vstack((to_return, v))
        i += 1
    return to_return
    
def get_idx_dataset(data,
                    vocab_lookup,
                    tag_lookup,
                    direction_lookup,
                    edge_lookup):
    tmp = list()
    for i in data:
        tmp.append(get_idx(i, vocab_lookup, tag_lookup, direction_lookup, edge_lookup))
    return tmp