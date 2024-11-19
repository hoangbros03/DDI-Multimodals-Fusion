import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from ddi_kt_2024.multimodal.bert import BertForFormulaClassification

from ddi_kt_2024.mol.preprocess import split_formula

class FormulaDataset(Dataset):
    def __init__(self, 
                x,
                lookup_character,
                split_option: str = '1',
                element: int = 1):
        self.lookup_character = lookup_character
        mols = list()
        if element == 1:
            for m in x:
                x1 = split_formula(m[0], option=split_option)
                lst = list()
                for c in x1:
                    if c in self.lookup_character.keys():
                      lst.append(self.lookup_character[c])
                    else:
                        lst.append(0)
                lst = torch.tensor(lst)
                mols.append(lst)
        elif element == 2:
            for m in x:
                x2 = split_formula(m[1], option=split_option)
                lst = list()
                for c in x2:
                    if c in self.lookup_character.keys():
                      lst.append(self.lookup_character[c])
                    else:
                        lst.append(0)
                lst = torch.tensor(lst)
                mols.append(lst)
            
        self.x = mols

    def batch_padding(self, batch_size):
        current = 0
        to_return = []
        while current + batch_size < len(self.x):
            batch = self.x[current:current+batch_size]
            max_len_in_batch = max([x.shape[0] for x in batch])

            for i in range(len(batch)):
                tmp = F.pad(batch[i], (0, max_len_in_batch - batch[i].shape[0]), "constant", 0)
                to_return.append(tmp)

            current += batch_size
        
        batch = self.x[current:]
        max_len_in_batch = max([x.shape[0] for x in batch])

        for i in range(len(batch)):
            tmp = F.pad(batch[i], (0, max_len_in_batch - batch[i].shape[0]), "constant", 0)
            to_return.append(tmp)

        self.x = to_return
      
    def negative_instance_filtering(self, path):
        with open(path, 'r') as f:
            lines = f.read().split('\n')[:-1]
            lst = [int(x.strip()) for x in lines]

        new_x = list()

        for idx in lst:
            new_x.append(self.x[idx])

        self.x = new_x

    def __getitem__(self, idx):
        x = self.x[idx]
        return x

    def __len__(self):
        return len(self.x)

class FormulaDataloader(Dataset):
    def __init__(self, 
                x,
                bert_model_path: str = 'allenai/scibert_scivocab_uncased',
                batch_size: int = 128,
                nit_path: str = None,
                split_option: str = '1',
                element: int = 1,
                device: str = 'cpu'):
      
        with open(nit_path, 'r') as f:
            lines = f.read().split('\n')[:-1]
            lst = [int(x.strip()) for x in lines]

        new_x = list()
        for idx in range(len(x)):
            if idx in lst:
                new_x.append(x[idx])

        x = new_x

        mols = list()

        bert_model = BertForFormulaClassification(model_name=bert_model_path)

        if element == 1:
            idx = 0
            while idx + batch_size < len(x):
                current_batch = x[idx:idx+batch_size]
                tmp = [' '.join(split_formula(m[0], option=split_option)) for m in current_batch]
                for i in range(len(tmp)):
                    if tmp[i] == 'None 1':
                        tmp[i] = ''
                tokens = bert_model.tokenizer(tmp, 
                                              return_tensors='pt', 
                                              padding=True).to(device)
                mols.append(tokens)
                idx += batch_size
            current_batch = x[idx:]
            tmp = [' '.join(split_formula(m[0], option=split_option)) for m in current_batch]
            for i in range(len(tmp)):
                if tmp[i] == 'None 1':
                    tmp[i] = ''
            tokens = bert_model.tokenizer(tmp, 
                                          return_tensors='pt', 
                                          padding=True).to(device)
            mols.append(tokens)
        elif element == 2:
            idx = 0
            while idx + batch_size < len(x):
                current_batch = x[idx:idx+batch_size]
                tmp = [' '.join(split_formula(m[1], option=split_option)) for m in current_batch]
                for i in range(len(tmp)):
                    if tmp[i] == 'None 1':
                        tmp[i] = ''
                tokens = bert_model.tokenizer(tmp, 
                                              return_tensors='pt', 
                                              padding=True).to(device)
                mols.append(tokens)
                idx += batch_size
            current_batch = x[idx:]
            tmp = [' '.join(split_formula(m[1], option=split_option)) for m in current_batch]
            for i in range(len(tmp)):
                if tmp[i] == 'None 1':
                    tmp[i] = ''
            tokens = bert_model.tokenizer(tmp, 
                                          return_tensors='pt', 
                                          padding=True).to(device)
            mols.append(tokens)
            
        self.x = mols

    def __getitem__(self, idx):
        x = self.x[idx]
        return x

    def __len__(self):
        return len(self.x)