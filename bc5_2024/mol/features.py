import torch
from rdkit import Chem
from torch_geometric.data import Data

def get_edge_index(mol):
  edges = []
  for bond in mol.GetBonds():
    i = bond.GetBeginAtomIdx()
    j = bond.GetEndAtomIdx()
    edges.extend([(i,j), (j,i)])

  edge_index = list(zip(*edges))
  return edge_index

def atom_feature(atom):
  return [atom.GetAtomicNum(), 
          atom.GetDegree(),
          atom.GetTotalDegree(),
          atom.GetNumImplicitHs(),
          atom.GetExplicitValence(),
          atom.GetImplicitValence(),
          atom.GetNumRadicalElectrons(),
          atom.GetFormalCharge(),
          atom.GetMass(),
          atom.GetIsAromatic()]

def bond_feature(bond):
  return [bond.GetBondType(), 
          bond.GetStereo(),
          bond.GetIsConjugated(),
          bond.GetIsAromatic()]

def smi_to_pyg(smi):
    if smi == 'None':
      return Data(edge_index=torch.LongTensor([(0, 0), (0, 0)]),
                  x=torch.FloatTensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 2]]),
                  edge_attr=torch.FloatTensor([[0, 0, 2, 2], [0, 0, 2, 2]]),
                  mol="None",
                  smiles="None")

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
      return None

    id_pairs = ((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds())
    atom_pairs = [z for (i, j) in id_pairs for z in ((i, j), (j, i))]

    bonds = (mol.GetBondBetweenAtoms(i, j) for (i, j) in atom_pairs)
    atom_features = [atom_feature(a) for a in mol.GetAtoms()]
    bond_features = [bond_feature(b) for b in bonds]

    edge_index = list(zip(*atom_pairs))
    if edge_index == []:
      edge_index = torch.LongTensor([(0, 0), (0, 0)])
      edge_attr = torch.FloatTensor([[0, 0, 2, 2], [0, 0, 2, 2]])
    else:
      edge_index = torch.LongTensor(edge_index)
      edge_attr = torch.FloatTensor(bond_features)

    return Data(edge_index=edge_index,
                x=torch.FloatTensor(atom_features),
                edge_attr=edge_attr,
                mol=mol,
                smiles=smi)

def smi_to_bert(smi, embedding_dict, embedding_size):
    if smi == 'None':
      return torch.zeros([1, embedding_size])

    if smi in embedding_dict.keys():
        return torch.tensor(embedding_dict[smi]).unsqueeze(dim=0)
    else:
        return torch.zeros([1, embedding_size])