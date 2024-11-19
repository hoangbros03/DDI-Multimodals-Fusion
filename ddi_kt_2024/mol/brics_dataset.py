from torch.utils.data import Dataset
from ddi_kt_2024.mol.features import smiles_to_brics

class BricsDataset(Dataset):
    def __init__(self,
                 x,
                 brics_set_path: str = 'cache/brics/brics.set.full.pkl',
                 element: int = 1):
        mols = list()
        if element == 1:
            for m in x:
                mols.append(smiles_to_brics(m[0],
                                            brics_set_path))
        elif element == 2:
            for m in x:
                mols.append(smiles_to_brics(m[1], 
                                            brics_set_path))
                
        self.x = mols

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