import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import logging

from bc5_2024.text.model.huggingface_model import get_model
from bc5_2024 import logging_config

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def fix_exception(self):
        i = 0
        while i < len(self.data):
            if self.data[i].shape[1] == 0:
                print(f"WARNING: Exception at data {i}")
                self.data[i] = torch.zeros((1, 1, 14), dtype=int)
            else:
                i += 1
    
    def rm_none(self):
        i = 0
        while i < len(self.data):
            if self.data[i] == None:
                self.data[i] = torch.zeros((1, 14))
            else:
                i += 1
                
    def squeeze(self):
        for i in range(len(self.data)):
            self.data[i] = self.data[i].squeeze()

    def average_position_embedding(self, col_pos=[2, 3, 10, 11]):
        for i in range(len(self.data)):
            length = self.data[i].shape[1]
            for j in col_pos:
                self.data[i][:, :, j] = self.data[i][:, :, j] / length
            

    def batch_padding(self, batch_size, min_batch_size=3, dataset='ddi'):
        if dataset == 'ddi':
            current = 0
            to_return = []

            while current + batch_size < len(self.data):
                batch = self.data[current:current+batch_size]
                max_len_in_batch = max(max([x.shape[1] for x in batch]), min_batch_size)

                for i in range(len(batch)):
                    tmp = F.pad(batch[i], (0, 0, 0, max_len_in_batch - batch[i].shape[1], 0, 0), "constant", 0)
                    to_return.append(tmp)

                current += batch_size

            batch = self.data[current:]
            max_len_in_batch = max(max([x.shape[0] for x in batch]), min_batch_size)

            for i in range(len(batch)):
                tmp = F.pad(batch[i], (0, 0, 0, max_len_in_batch - batch[i].shape[1], 0, 0), "constant", 0)
                to_return.append(tmp)

            self.data = to_return
        elif dataset == 'bc5':
            current = 0
            to_return = []

            while current + batch_size < len(self.data):
                batch = self.data[current:current+batch_size]

                max_len_in_batch = max(max([x.shape[0] for x in batch]), min_batch_size)

                for i in range(len(batch)):
                    tmp = F.pad(batch[i], (0, 0, 0, max_len_in_batch - batch[i].shape[0]), "constant", 0)
                    to_return.append(tmp)

                current += batch_size

            batch = self.data[current:]
            max_len_in_batch = max(max([x.shape[0] for x in batch]), min_batch_size)

            for i in range(len(batch)):
                tmp = F.pad(batch[i], (0, 0, 0, max_len_in_batch - batch[i].shape[0]), "constant", 0)
                to_return.append(tmp)

            self.data = to_return

    def rm_no_smiles(self, list_index):
        new_x = list()
        new_y = list()
        for idx in list_index:
            new_x.append(self.data[idx])
            new_y.append(self.labels[idx])

        self.data = new_x
        self.labels = new_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
            sample = self.data[idx]
            label = self.labels[idx]
            return sample, label