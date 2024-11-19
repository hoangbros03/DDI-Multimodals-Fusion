from torch.utils.data import Dataset
from tqdm import tqdm
from ddi_kt_2024.mol.features import smiles_to_pyg

class MolDataset(Dataset):
  def __init__(self, x, element=1):
    mols = list()
    if element == 1:
      for m in tqdm(x, desc="Converting SMILES to PyG"):
        x1 = smiles_to_pyg(m[0])
        mols.append(x1)
    elif element == 2:
      for m in tqdm(x, desc="Converting SMILES to PyG"):
        x2 = smiles_to_pyg(m[1])
        mols.append(x2)
    self.x = [m for m in mols]

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