from torch.utils.data import Dataset

class MultimodalDataset(Dataset):
  def __init__(self, text_dataset, mol_dataset):
    data_list = list()
    label = list()
    for i in range(len(text_dataset)):
      data_list.append([text_dataset[i][0], mol_dataset[i][0]])
      label.append(mol_dataset[i][1])
    self.x = data_list
    self.y = label

  def __getitem__(self, idx):
    x = self.x[idx]
    y = self.y[idx]
    return x, y

  def __len__(self):
    return len(self.x)