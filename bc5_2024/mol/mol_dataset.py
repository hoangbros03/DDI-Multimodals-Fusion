from torch.utils.data import Dataset
from tqdm import tqdm
from bc5_2024.mol.features import smi_to_pyg
from torch_geometric.loader import DataLoader

class MolDataset(Dataset):
  def __init__(self, x):
    mols = list()
    for m in tqdm(x, desc="Converting SMILES to PyG"):
      x1 = smi_to_pyg(m[0])
      mols.append(x1)
    self.x = [m for m in mols]

  def __getitem__(self, idx):
    x = self.x[idx]
    return x

  def __len__(self):
    return len(self.x)