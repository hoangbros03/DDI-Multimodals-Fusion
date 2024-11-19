from torch.utils.data import Dataset
from ddi_kt_2024.multimodal.bert import BertForSmilesClassification

class SmilesDataloader(Dataset):
  def __init__(self, 
               x,
               bert_model: BertForSmilesClassification,
               max_length: int = 64,
               batch_size: int = 128,
               nit_path: str = None,
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

    if element == 1:
        idx = 0
        while idx + batch_size < len(x):
            current_batch = x[idx:idx+batch_size]
            tmp = [x[0] for x in current_batch]
            tokens = bert_model.tokenizer(tmp, 
                                          return_tensors='pt', 
                                          padding=True,
                                          truncation=True,
                                          max_length=max_length).to(device)
            mols.append(tokens)
            idx += batch_size
        current_batch = x[idx:]
        tmp = [x[0] for x in current_batch]
        tokens = bert_model.tokenizer(tmp, 
                                      return_tensors='pt', 
                                      padding=True,
                                      truncation=True,
                                      max_length=max_length).to(device)
        mols.append(tokens)
    elif element == 2:
        idx = 0
        while idx + batch_size < len(x):
            current_batch = x[idx:idx+batch_size]
            tmp = [x[1] for x in current_batch]
            tokens = bert_model.tokenizer(tmp, 
                                          return_tensors='pt', 
                                          padding=True,
                                          truncation=True,
                                          max_length=max_length).to(device)
            mols.append(tokens)
            idx += batch_size
        current_batch = x[idx:]
        tmp = [x[1] for x in current_batch]
        tokens = bert_model.tokenizer(tmp, 
                                      return_tensors='pt', 
                                      padding=True,
                                      truncation=True,
                                      max_length=max_length).to(device)
        mols.append(tokens)
        
    self.x = mols

  def __getitem__(self, idx):
      x = self.x[idx]
      return x

  def __len__(self):
      return len(self.x)