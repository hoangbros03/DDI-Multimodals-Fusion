"""
Similar method of asada's paper
Details: https://github.com/tticoin/DESC_MOL-DDIE
"""
import ast

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BertModel
from transformers.data.processors.utils import InputExample, InputFeatures, DataProcessor
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import glue_output_modes as output_modes

from ddi_kt_2024.utils import load_pkl

def fix_candidates(candidates, can_type):
    # Get dict
    if can_type == "train":
        file_path = 'cache/drugbank/correct_ddi_train.txt'
    else:
        file_path = 'cache/drugbank/correct_ddi_test.txt'
    data = {}
    ids = []
    with open(file_path, 'r') as file:
        new_id = 2
        current_id = None
        for line in file:
            line = line.strip()
            if line=="":
                new_id=2
                continue
            if new_id==2:
                current_id = line
                ids.append(line)
                data[current_id] = {}
                new_id = 1
                continue
            elif new_id==1:
                data[current_id]['text'] = line
                data[current_id]['entities'] = {}
                new_id=0
                continue
            entitiy_dict =ast.literal_eval(line)
            data[current_id]['entities'][entitiy_dict['@id']] = {
                '@charOffset': entitiy_dict['@charOffset'],
                '@text': entitiy_dict['@text']
            }

    # Fix candidates
    for idx, candidate in enumerate(candidates):
        c_id = ".".join(candidate['id'].split(".")[:-1])
        if c_id in ids:
            # Get entities id
            e1_id = candidate['e1']['@id']
            e2_id = candidate['e2']['@id']
            candidates[idx]['text'] = data[c_id]['text']
            candidates[idx]['e1']['@text'] = data[c_id]['entities'][e1_id]['@text']
            candidates[idx]['e1']['@charOffset'] = data[c_id]['entities'][e1_id]['@charOffset']
            candidates[idx]['e2']['@text'] = data[c_id]['entities'][e2_id]['@text']
            candidates[idx]['e2']['@charOffset'] = data[c_id]['entities'][e2_id]['@charOffset']
    print("Fix complete!")
    return candidates
            
def convert_to_examples(candidates, can_type="train", mask=True, save_path=None, data_type='ddi', trim=-1, desc_type="origin"):
    """
    Return:
    An InputExample with the following structure:
    {
        "guid": id of candidate,
        "label": label of its candidate,
        "text_a": text,
        "text_b": always null 
    }
     examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    """
    # Change with DRUGOTHER and DRUG1, DRUG2
    current_sentence = ""
    examples = []
    lb_list = ['false', 'advise', 'effect',  'mechanism', 'int']

    if data_type == "desc":
        examples_desc_1 = []
        examples_desc_2 = []
        try:
            if desc_type == "origin":
                print("Choosing desc_type origin")
                desc_dict = load_pkl("cache/desc_dict/desc_dict_origin.pkl")
            elif desc_type == "full":
                print("Choosing desc_type full")
                desc_dict = load_pkl("cache/desc_dict/desc_dict_full.pkl")
            else:
                print("Choosing desc_type normal. Please check because it reduce the F1 result")
                desc_dict = load_pkl("cache/desc_dict/desc_dict.pkl")
        except Exception as e:
            print(f"Error when loading desc_dict: {e}")
            return
    
    if data_type != "desc":
        # Fix the candidates
        candidates = fix_candidates(candidates, can_type=can_type)

    for idx, candidate in tqdm(enumerate(candidates), total=len(candidates)):
        # HANDLE DESCRIPTION
        if data_type == 'desc':
            # Get entities, lower case
            e1_text = candidate['e1']['@text'].lower()
            e2_text = candidate['e2']['@text'].lower()
            
            # Map to desc
            # In here, if we can't find the entity, at least we know the name of the entity
            if e1_text in desc_dict:
                if desc_dict[e1_text] is not None:
                    e1_text = desc_dict[e1_text]
            if e2_text in desc_dict:
                if desc_dict[e2_text] is not None:
                    e2_text = desc_dict[e2_text]
            
            # Create InputExample
            examples_desc_1.append(
                InputExample(guid=f"{can_type}_{idx+1}", text_a=str(e1_text), text_b="", label=candidate['label'])
            )
            examples_desc_2.append(
                InputExample(guid=f"{can_type}_{idx+1}", text_a=str(e2_text), text_b="", label=candidate['label'])
            )
            continue
           
        # HANDLE NORMAL
        if candidate['text'] != current_sentence:
            current_sentence = candidate['text']
            all_sentence_entities = set()
            inc = 1
            while True:
                if len(candidates) <= idx + inc:
                    break
                if candidates[idx+inc]['text'] != current_sentence:
                    break
                all_sentence_entities.add(candidates[idx+inc]['e1']['@text'])
                all_sentence_entities.add(candidates[idx+inc]['e2']['@text'])
                inc +=1
            
            all_sentence_entities=list(all_sentence_entities)
        
        offset_1 = candidate['e1']['@charOffset']
        # offset_1 = offset_1.split(';')[0]
        offset_1 = (int(offset_1.split('-')[0]), int(offset_1.split('-')[1]))
        
        offset_2 = candidate['e2']['@charOffset']
        # offset_2 = offset_2.split(';')[0]
        offset_2 = (int(offset_2.split('-')[0]), int(offset_2.split('-')[1]))

        if mask:
            current_sentence = current_sentence[:offset_1[0]] + "DRUG1" + current_sentence[offset_1[1]+1:]
            if offset_1[1] < offset_2[0]:
                diff_len = len(candidate['e1']['@text']) - 5
                current_sentence = current_sentence[:offset_2[0]-diff_len] + "DRUG2" + current_sentence[offset_2[1] +1 - diff_len:]
            else:
                current_sentence = current_sentence[:offset_2[0]] + "DRUG2" + current_sentence[offset_2[1]+1:]
            for entity in all_sentence_entities:
                current_sentence = current_sentence.replace(entity, "DRUGOTHER")
        
        if trim >= 0:
            _cs = current_sentence
            p1 = current_sentence.find("DRUG1")
            p2 = current_sentence.find("DRUG2")
            p1, p2 = min(p1,p2), max(p1, p2)

            if trim==0:
                current_sentence = current_sentence[p1:p2+5]
            else:
                # Find offset if extend 1, 2...
                s_partition = current_sentence.split(" ")
                for ele in s_partition:
                    if ele.find("DRUG1") != -1:
                        offset_1 = s_partition.index(ele)
                    if ele.find("DRUG2") != -1:
                        offset_2 = s_partition.index(ele)
                try:
                    offset_1, offset_2 = min(offset_1, offset_2), max(offset_1, offset_2)
                    offset_1 = max(0, offset_1 - trim)
                    offset_2 = min(len(s_partition)-1, offset_2 + trim)
                    current_sentence = " ".join(s_partition[offset_1:offset_2+1])
                except Exception as e:
                    print(f"Idx: {idx}, Exceptional case, no worry...")
                    current_sentence = _cs
        if data_type == 'ddi':
            examples.append(
                InputExample(guid=f"{can_type}_{idx+1}", text_a=current_sentence, text_b="", label=candidate['label'])
            )
        elif data_type == "bc5":
            examples.append(
                InputExample(guid=f"{can_type}_{idx+1}", text_a=current_sentence, text_b="", label=lb_list[candidate['label']])
            )
    
    # Save as tuple
    if data_type=="desc":
        examples = tuple((examples_desc_1, examples_desc_2))
    
    if save_path is not None:
        torch.save(examples, save_path)
    return examples

def new_convert_to_examples(candidates, can_type="train", mask=True, save_path=None, data_type='ddi', trim=-1, desc_type="origin", drugother_mask="full"):
    """
    Return:
    An InputExample with the following structure:
    {
        "guid": id of candidate,
        "label": label of its candidate,
        "text_a": text,
        "text_b": always null 
    }
     examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    """
    # Change with DRUGOTHER and DRUG1, DRUG2
    print("Drugother mask: ", drugother_mask)
    current_sentence = ""
    examples = []
    lb_list = ['false', 'advise', 'effect',  'mechanism', 'int']

    if data_type == "desc":
        examples_desc_1 = []
        examples_desc_2 = []
        try:
            if desc_type == "origin":
                print("Choosing desc_type origin")
                desc_dict = load_pkl("cache/desc_dict/desc_dict_origin.pkl")
            elif desc_type == "full":
                print("Choosing desc_type full")
                desc_dict = load_pkl("cache/desc_dict/desc_dict_full.pkl")
            else:
                print("Choosing desc_type normal. Please check because it reduce the F1 result")
                desc_dict = load_pkl("cache/desc_dict/desc_dict.pkl")
        except Exception as e:
            print(f"Error when loading desc_dict: {e}")
            return
    
    if data_type != "desc":
        # Fix the candidates
        candidates = fix_candidates(candidates, can_type=can_type)

    for idx, candidate in tqdm(enumerate(candidates), total=len(candidates)):
        # HANDLE DESCRIPTION
        if data_type == 'desc':
            # Get entities, lower case
            e1_text = candidate['e1']['@text'].lower()
            e2_text = candidate['e2']['@text'].lower()
            
            # Map to desc
            # In here, if we can't find the entity, at least we know the name of the entity
            if e1_text in desc_dict:
                if desc_dict[e1_text] is not None:
                    e1_text = desc_dict[e1_text]
            if e2_text in desc_dict:
                if desc_dict[e2_text] is not None:
                    e2_text = desc_dict[e2_text]
            
            # Create InputExample
            examples_desc_1.append(
                InputExample(guid=f"{can_type}_{idx+1}", text_a=str(e1_text), text_b="", label=candidate['label'])
            )
            examples_desc_2.append(
                InputExample(guid=f"{can_type}_{idx+1}", text_a=str(e2_text), text_b="", label=candidate['label'])
            )
            continue
           
        # HANDLE NORMAL
        if candidate['text'] != current_sentence:
            current_sentence = candidate['text']
            _cs = current_sentence
            all_sentence_entities = set()
            inc = 1
            while True:
                if len(candidates) <= idx + inc:
                    break
                if candidates[idx+inc]['text'] != current_sentence:
                    break
                all_sentence_entities.add(candidates[idx+inc]['e1']['@text'])
                all_sentence_entities.add(candidates[idx+inc]['e2']['@text'])
                inc +=1
            
            all_sentence_entities=list(all_sentence_entities)
        
        offset_1 = candidate['e1']['@charOffset']
        # offset_1 = offset_1.split(';')[0]
        offset_1 = (int(offset_1.split('-')[0]), int(offset_1.split('-')[1]))
        
        offset_2 = candidate['e2']['@charOffset']
        # offset_2 = offset_2.split(';')[0]
        offset_2 = (int(offset_2.split('-')[0]), int(offset_2.split('-')[1]))

        if mask:
            current_sentence = current_sentence[:offset_1[0]] + "DRUG1" + current_sentence[offset_1[1]+1:]
            if offset_1[1] < offset_2[0]:
                diff_len = len(candidate['e1']['@text']) - 5
                current_sentence = current_sentence[:offset_2[0]-diff_len] + "DRUG2" + current_sentence[offset_2[1] +1 - diff_len:]
            else:
                current_sentence = current_sentence[:offset_2[0]] + "DRUG2" + current_sentence[offset_2[1]+1:]
            if drugother_mask=="right_drugs":
                min_mask_index = min(current_sentence.find("DRUG1"), current_sentence.find("DRUG2"))
                for entity in all_sentence_entities:
                    if current_sentence.find(entity)>min_mask_index:
                        current_sentence = current_sentence.replace(entity, "DRUGOTHER")
            elif drugother_mask=="full":
                for entity in all_sentence_entities:
                    current_sentence = current_sentence.replace(entity, "DRUGOTHER")
            elif drugother_mask=="between":
                _idxs = [current_sentence.find("DRUG1"), current_sentence.find("DRUG1")+5,current_sentence.find("DRUG2"), current_sentence.find("DRUG2")+5]
                min_mask_index, max_mask_index = min(_idxs), max(_idxs)
                for entity in all_sentence_entities:
                    if current_sentence.find(entity)>min_mask_index and current_sentence.find(entity) < max_mask_index:
                        current_sentence = current_sentence.replace(entity, "DRUGOTHER")
            elif drugother_mask=="left_drugs":
                max_mask_index = max(current_sentence.find("DRUG1")+5, current_sentence.find("DRUG2")+5)
                for entity in all_sentence_entities:
                    if current_sentence.find(entity)<max_mask_index:
                        current_sentence = current_sentence.replace(entity, "DRUGOTHER")
            elif drugother_mask=="outbound":
                _idxs = [current_sentence.find("DRUG1"), current_sentence.find("DRUG1")+5,current_sentence.find("DRUG2"), current_sentence.find("DRUG2")+5]
                min_mask_index, max_mask_index = min(_idxs), max(_idxs)
                for entity in all_sentence_entities:
                    if current_sentence.find(entity)<min_mask_index or current_sentence.find(entity) > max_mask_index:
                        current_sentence = current_sentence.replace(entity, "DRUGOTHER")
        if trim >= 0:
            _cs = current_sentence
            p1 = current_sentence.find("DRUG1")
            p2 = current_sentence.find("DRUG2")
            p1, p2 = min(p1,p2), max(p1, p2)

            if trim==0:
                current_sentence = current_sentence[p1:p2+5]
            else:
                # Find offset if extend 1, 2...
                s_partition = current_sentence.split(" ")
                for ele in s_partition:
                    if ele.find("DRUG1") != -1:
                        offset_1 = s_partition.index(ele)
                    if ele.find("DRUG2") != -1:
                        offset_2 = s_partition.index(ele)
                try:
                    offset_1, offset_2 = min(offset_1, offset_2), max(offset_1, offset_2)
                    offset_1 = max(0, offset_1 - trim)
                    offset_2 = min(len(s_partition)-1, offset_2 + trim)
                    current_sentence = " ".join(s_partition[offset_1:offset_2+1])
                except Exception as e:
                    print(f"Idx: {idx}, Exceptional case, no worry...")
                    current_sentence = _cs
        if data_type == 'ddi':
            examples.append(
                InputExample(guid=f"{can_type}_{idx+1}", text_a=current_sentence, text_b="", label=candidate['label'])
            )
        elif data_type == "bc5":
            examples.append(
                InputExample(guid=f"{can_type}_{idx+1}", text_a=current_sentence, text_b="", label=lb_list[candidate['label']])
            )
        current_sentence = _cs
    # Save as tuple
    if data_type=="desc":
        examples = tuple((examples_desc_1, examples_desc_2))
    
    if save_path is not None:
        torch.save(examples, save_path)
    return examples

def preprocess(examples, model_name, max_seq_length=128, save_path=None, desc = False, can_type='train'):
    if isinstance(examples, str):
        examples = torch.load(examples)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = BertModel.from_pretrained(model_name)
    
    # Normal
    if not desc:
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=['false', 'advise', 'effect',  'mechanism', 'int'],
            max_length=max_seq_length,
            output_mode=output_modes['mrpc'],
        )

        # Get Position index
        drug_id = tokenizer.vocab['drug']
        one_id = tokenizer.vocab['##1']
        two_id = tokenizer.vocab['##2']

        all_input_ids = [f.input_ids for f in features]

        all_entity1_pos= []
        all_entity2_pos= []
        for input_ids in all_input_ids:
            entity1_pos = max_seq_length-1 
            entity2_pos = max_seq_length-1 
            for i in range(max_seq_length):
                if input_ids[i] == drug_id and input_ids[i+1] == one_id:
                    entity1_pos = i
                if input_ids[i] == drug_id and input_ids[i+1] == two_id:
                    entity2_pos = i
            all_entity1_pos.append(entity1_pos)
            all_entity2_pos.append(entity2_pos)

        range_list = list(range(max_seq_length, 2*max_seq_length))
        all_relative_dist1 = torch.tensor([[x - e1 for x in range_list] for e1 in all_entity1_pos], dtype=torch.long)
        all_relative_dist2 = torch.tensor([[x - e2 for x in range_list] for e2 in all_entity2_pos], dtype=torch.long)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

        dataset = TensorDataset(
            all_input_ids, all_attention_mask, all_token_type_ids,
            all_relative_dist1, all_relative_dist2,
            all_labels)

    else: # Description
        examples_1, examples_2 = examples
        features_1 = convert_examples_to_features(
            examples_1,
            tokenizer,
            label_list=['false', 'advise', 'effect',  'mechanism', 'int'],
            max_length=max_seq_length,
            output_mode=output_modes['mrpc'],
        )
        features_2 = convert_examples_to_features(
            examples_2,
            tokenizer,
            label_list=['false', 'advise', 'effect',  'mechanism', 'int'],
            max_length=max_seq_length,
            output_mode=output_modes['mrpc'],
        )
        all_input_ids_1 = torch.tensor([f.input_ids for f in features_1], dtype=torch.long)
        all_input_ids_2 = torch.tensor([f.input_ids for f in features_2], dtype=torch.long)
        all_attention_mask_1 = torch.tensor([f.attention_mask for f in features_1], dtype=torch.long)
        all_attention_mask_2 = torch.tensor([f.attention_mask for f in features_2], dtype=torch.long)
        all_token_type_ids_1 = torch.tensor([f.token_type_ids for f in features_1], dtype=torch.long)
        all_token_type_ids_2 = torch.tensor([f.token_type_ids for f in features_2], dtype=torch.long)

        dataset = TensorDataset(all_input_ids_1, all_attention_mask_1, all_token_type_ids_1, all_input_ids_2, all_attention_mask_2, all_token_type_ids_2)
        dataset = _negative_filtering(dataset, data_type=can_type)

    if save_path is not None:
        torch.save(dataset, save_path)
    
    return dataset

def _negative_filtering(dataset, data_type="train"):
    if data_type == "train":
        txt_path = "cache/filtered_ddi/train_filtered_index.txt"
    elif data_type == "test":
        txt_path = "cache/filtered_ddi/test_filtered_index.txt"
    else:
        print("Wrong prepare_type, only support train and test")
        return
    with open(txt_path, "r") as f:
        lines = f.read().split('\n')[:-1]
        filtered_idx = [int(x.strip()) for x in lines]

    new_data = [list() for _ in range(len(dataset.tensors))]
    for idx in filtered_idx:
        for t_idx in range(len(new_data)):
            new_data[t_idx].append(dataset.tensors[t_idx][idx])
    torch.save(new_data, "new_data.pt")

    new_dataset = TensorDataset(
        *[torch.tensor(torch.stack(new_data[i], dim=0), dtype=torch.long) for i in range(len(new_data))]
    )
    
    print("Negative filtering ok!")
    return new_dataset
