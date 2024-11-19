import torch
from torch.utils.data import Dataset
from ddi_kt_2024.mol.features import smiles_to_fn_group

class FnGroupDataset(Dataset):
    def __init__(self, x):
        mols = list()
        for m in x:
            fng1 = smiles_to_fn_group(m[0])
            fng2 = smiles_to_fn_group(m[1])
            cooc = self.cooccurrence(fng1, fng2)
            mols.append(cooc)
                
        self.x = mols

    def cooccurrence(self, fng1, fng2):
        lst = list()
        length = fng1.shape[1]
        for i in range(length):
            for j in range(i, length):
                if fng1[0][i] > 0 and fng2[0][j] > 0:
                    lst.append(1)
                else:
                    lst.append(0)
        return torch.FloatTensor(lst).unsqueeze(dim=0)

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