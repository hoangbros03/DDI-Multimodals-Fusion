import pickle as pkl
from pathlib import Path
import logging

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

import ddi_kt_2024.logging_config
from ddi_kt_2024.text.reader.yaml_reader import save_yaml_config

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
        if candidate['label'] == 'false':
            label_list.append(0)
        elif candidate['label'] == 'advise':
            label_list.append(1)
        elif candidate['label'] == 'effect':
            label_list.append(2)
        elif candidate['label'] == 'mechanism':
            label_list.append(3)
        elif candidate['label'] == 'int':
            label_list.append(4)
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

def save_model(output_path, file_name, config, model, wandb_available=False, pred_logit=None):
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
    dump_pkl(pred_logit, str(Path(output_path) / f"pred_logit_{str(file_name[:-3])}.pkl"))
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

def read_index(path):
    """
    Read filtered index in DDI.
    """
    with open(path, 'r') as f:
        lines = f.read().split('\n')[:-1]
        lst = [int(x.strip()) for x in lines]
    return lst

def convert_to_label_list(custom_dataset):
    """
    Convert label in CustomDataset to list.
    """
    lst = list()
    for x in custom_dataset:
        lst.append(int(x[1].cpu().numpy()[0]))
        
    return lst

def get_medline_and_drugbank_result(pred):
    """Pred shape: (3208,) or (3208,5)"""
    def ddie_compute_metrics(preds, labels, every_type=True):
        label_list = ('Advise', 'Effect', 'Mechanism', 'Int.')
        p, r, f, s = precision_recall_fscore_support(y_pred=preds, y_true=labels, labels=[1,2,3,4], average='micro')
        result = {
            "Precision": p,
            "Recall": r,
            "microF": f
        }
        p, r, f, s = precision_recall_fscore_support(y_pred=preds, y_true=labels, labels=[1,2,3,4], average='macro')
        result['macroF'] = f
        if every_type:
            evaluation = precision_recall_fscore_support(y_pred=preds, y_true=labels, labels=[1,2,3,4], average=None)
            for i, label_type in enumerate(label_list):
                result[label_type + ' Precision'] = evaluation[0][i]
                result[label_type + ' Recall'] = evaluation[1][i]
                result[label_type + ' F'] = evaluation[2][i]
        return result
    
    all_candidates_test = load_pkl('cache/pkl/v2/notprocessed.candidates.test.pkl')
    full_labels = get_labels(all_candidates_test)
    with open('cache/filtered_ddi/test_filtered_index.txt', 'r') as f:
        lines = f.read().split('\n')[:-1]
        filtered_lst_index_test = [int(x.strip()) for x in lines]
    full_labels_filtered = []
    for i in range(len(full_labels)):
        if i in filtered_lst_index_test:
            full_labels_filtered.append(full_labels[i])
    if len(list(pred.shape))==2:
        pred = torch.argmax(pred, dim=-1)
    return {
        'drugbank': ddie_compute_metrics(pred[:2705], full_labels_filtered[:2705]),
        'medline': ddie_compute_metrics(pred[2705:], full_labels_filtered[2705:])
    }

from pathlib import Path
from ddi_kt_2024.multimodal.all_multimodal import *

def list_folders(directory):
    return [folder.absolute() for folder in Path(directory).glob('*') if folder.is_dir()]

def get_result_from_checkpoint(dir_path, tuple_dataloader):
    """
    Get result from checkpoint in a folder
    
    """
    dataloader_text_test, dataloader_test_graph1, dataloader_test_graph2, dataloader_test_for1, dataloader_test_for2, dataloader_test_desc1, dataloader_test_desc2, dataloader_image_test = tuple_dataloader
    def _get_result(state_dict_path):
        s_dict = torch.load(state_dict_path)
        kwargs = {
        'seed': 42,
        'desc_conv_output_size': s_dict['norm_desc.weight'].shape[0],
        'desc_conv_window_size': 3,
        'desc_layer_hidden': 0,
        'formula_conv_output_size': s_dict['norm_formula.weight'].shape[0],
        'formula_conv_window_size': 4,
        'freeze_gnn': True,
        'freeze_formula': True,
        'gnn_option': 'GCN',
        'readout_option': 'global_max_pool',
        'num_layers_gnn': 5,
        'hidden_channels': s_dict['norm_graph.weight'].shape[0],
        'use_gat_v2': False,
        'use_edge_attr': True,
        }
        if 'middle_classifier.weight' in s_dict.keys():
            middle_layer_size = s_dict['middle_classifier.weight'].shape[0]
        else:
            middle_layer_size=0

        model = BertForSequenceClassification(num_labels=5,
                    dropout_rate_other=0.1,
                    dropout_rate_bert_output=0.1,
                    out_conv_list_dim=128,
                    conv_window_size=[3,5,7],
                    max_seq_length=256,
                    middle_layer_size=middle_layer_size,
                    model_name_or_path='/kaggle/input/cp-79-8',
                    pos_emb_dim=s_dict['pos_emb.weight'].shape[1],
                    activation="gelu",
                    weight_decay=0.0,
                    freeze_bert=True,
                    data_type="ddi_no_negative",
                    config=BertConfig.from_pretrained('/kaggle/input/cp-79-8'),
                    image_reduced_dim=s_dict['batch_norm_img.weight'].shape[0],
                    apply_mask=False,
                    **kwargs
                    )
        logits_result = torch.Tensor([])
        validation_loader=dataloader_text_test
        test_mol1_loader=dataloader_test_graph1
        test_mol2_loader=dataloader_test_graph2
        test_mol3_loader=dataloader_test_for1
        test_mol4_loader=dataloader_test_for2
        test_mol5_loader=dataloader_test_desc1
        test_mol6_loader=dataloader_test_desc2
        test_image_loader=dataloader_image_test
        modal1='graph'
        modal2='formula'
        modal3='desc'
        model.load_state_dict(s_dict, strict=True)
        all_modal_dataloader = zip(validation_loader, test_mol1_loader, test_mol2_loader, test_mol3_loader, test_mol4_loader, test_mol5_loader, test_mol6_loader, test_image_loader)
        for (batch, mol1, mol2, mol3, mol4, mol5, mol6, image) in tqdm(all_modal_dataloader):
            model.eval()
            batch = tuple(t.to('cuda') for t in batch)
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2],
                        'relative_dist1': batch[3],
                        'relative_dist2': batch[4],
                        'labels':         batch[5],
                        f'{modal1}1': mol1.to('cuda'),
                        f'{modal1}2': mol2.to('cuda'),
                        f'{modal2}1': mol3.to('cuda'),
                        f'{modal2}2': mol4.to('cuda'),
                        f'{modal3}1': mol5.to('cuda'),
                        f'{modal3}2': mol6.to('cuda'),
                        'image_classifier_output1': image[5].to('cuda')}
                model.eval()
                result = nn.Softmax(dim=1)(model.cuda()(**inputs)[1]).to('cpu')
            del inputs
            del batch
            del mol1
            del mol2
            del mol3
            del mol4
            del mol5
            del mol6
            del image
            logits_result = torch.cat((logits_result, result), dim=0)
            del result
        return logits_result
            
    directory = Path(dir_path)
    folders = list_folders(directory)
    all_results = torch.Tensor([])

    # Scan though folders, work with dataset in https://www.kaggle.com/datasets/tantantankiw/checkpoint-htb-all-0-1-2-3
    for f_idx, folder in enumerate(list(folders)):
        print(f"{f_idx+1}/{len(list(folders))}")
        state_dict_path=[file.absolute() for file in Path(folder).glob('*.pt')][0]
        try:
            logits_result = _get_result(state_dict_path)
        except Exception as e:
            print(e)
            print("Continue...")
            continue
        all_results = torch.cat((all_results, logits_result.unsqueeze_(dim=0).to('cpu')), dim=0).to('cpu')

    # Check every file inside this dir
    for f_idx, file in enumerate(list([file.absolute() for file in Path(folder).glob('*.pt')])):
        print(f"{f_idx+1}/{len(list([file.absolute() for file in Path(folder).glob('*.pt')]))}")
        try:
            logits_result = _get_result(file)
        except Exception as e:
            print(e)
            print("Continue...")
            continue
        all_results = torch.cat((all_results, logits_result.unsqueeze_(dim=0).to('cpu')), dim=0).to('cpu')

    return all_results
