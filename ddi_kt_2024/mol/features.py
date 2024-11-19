import torch
import rdkit
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.Fragments import *
from torch_geometric.data import Data
from ddi_kt_2024.utils import load_pkl
from collections import OrderedDict

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
            atom.GetNumImplicitHs(),
            atom.GetExplicitValence(),
            atom.GetImplicitValence(),
            atom.GetTotalValence(),
            atom.GetNumRadicalElectrons(),
            atom.GetHybridization(),
            atom.GetIsAromatic(),
            atom.IsInRing()]

def bond_feature(bond):
    return [bond.GetBondType(), 
            bond.GetStereo(),
            bond.GetIsConjugated(),
            bond.GetIsAromatic(),
            bond.IsInRing()]

def smiles_to_pyg(smi):
    """
    Convert SMILES to graph
    """
    if smi == 'None':
        return Data(edge_index=torch.LongTensor([[0], [0]]),
                    x=torch.FloatTensor([[0, 0, 0, 0, 0, 0, 0, 0, 2, 2]]),
                    edge_attr=torch.FloatTensor([[0, 0, 2, 2, 2]]),
                    mol="None",
                    smiles="None")

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return Data(edge_index=torch.LongTensor([[0], [0]]),
                    x=torch.FloatTensor([[0, 0, 0, 0, 0, 0, 0, 0, 2, 2]]),
                    edge_attr=torch.FloatTensor([[0, 0, 2, 2, 2]]),
                    mol="None",
                    smiles="None")

    id_pairs = ((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds())
    atom_pairs = [z for (i, j) in id_pairs for z in ((i, j), (j, i))]

    bonds = (mol.GetBondBetweenAtoms(i, j) for (i, j) in atom_pairs)
    atom_features = [atom_feature(a) for a in mol.GetAtoms()]
    bond_features = [bond_feature(b) for b in bonds]

    edge_index = list(zip(*atom_pairs))
    if edge_index == []:
        edge_index = torch.LongTensor([[0], [0]])
        edge_attr = torch.FloatTensor([[0, 0, 2, 2, 2]])
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
    
def smiles_to_brics(smiles, brics_set_path):
    """
    Convert SMILES to BRICS encoding
    """
    brics_set = load_pkl(brics_set_path)

    if smiles == 'None':
        return torch.zeros([1, len(brics_set)])
    
    m = Chem.MolFromSmiles(smiles)
    
    if m is None:
        return torch.zeros([1, len(brics_set)])
    
    frags = list(Chem.BRICS.BRICSDecompose(m, minFragmentSize=1))
    
    dct = OrderedDict()
    
    for b in brics_set:
        dct[b] = 0
    
    for brics in frags:
        if brics in brics_set:
            dct[brics] = 1
    
    return torch.FloatTensor(list(dct.values())).unsqueeze(dim=0)

def smiles_to_fn_group(smiles):
    """
    Convert SMILES to Functional Group encoding
    """
    if smiles == 'None':
        return torch.zeros([1, 85])
    
    m = Chem.MolFromSmiles(smiles)
    
    if m is None:
        return torch.zeros([1, 85])
    
    f_gr = [rdkit.Chem.Fragments.fr_Al_COO(m),
            rdkit.Chem.Fragments.fr_Al_OH(m),
            rdkit.Chem.Fragments.fr_Al_OH_noTert(m),
            rdkit.Chem.Fragments.fr_ArN(m),
            rdkit.Chem.Fragments.fr_Ar_COO(m),
            rdkit.Chem.Fragments.fr_Ar_N(m),
            rdkit.Chem.Fragments.fr_Ar_NH(m),
            rdkit.Chem.Fragments.fr_Ar_OH(m),
            rdkit.Chem.Fragments.fr_COO(m),
            rdkit.Chem.Fragments.fr_COO2(m),
            rdkit.Chem.Fragments.fr_C_O(m),
            rdkit.Chem.Fragments.fr_C_O_noCOO(m),
            rdkit.Chem.Fragments.fr_C_S(m),
            rdkit.Chem.Fragments.fr_HOCCN(m),
            rdkit.Chem.Fragments.fr_Imine(m),
            rdkit.Chem.Fragments.fr_NH0(m),
            rdkit.Chem.Fragments.fr_NH1(m),
            rdkit.Chem.Fragments.fr_NH2(m),
            rdkit.Chem.Fragments.fr_N_O(m),
            rdkit.Chem.Fragments.fr_Ndealkylation1(m),
            rdkit.Chem.Fragments.fr_Ndealkylation2(m),
            rdkit.Chem.Fragments.fr_Nhpyrrole(m),
            rdkit.Chem.Fragments.fr_SH(m),
            rdkit.Chem.Fragments.fr_aldehyde(m),
            rdkit.Chem.Fragments.fr_alkyl_carbamate(m),
            rdkit.Chem.Fragments.fr_alkyl_halide(m),
            rdkit.Chem.Fragments.fr_allylic_oxid(m),
            rdkit.Chem.Fragments.fr_amide(m),
            rdkit.Chem.Fragments.fr_amidine(m),
            rdkit.Chem.Fragments.fr_aniline(m),
            rdkit.Chem.Fragments.fr_aryl_methyl(m),
            rdkit.Chem.Fragments.fr_azide(m),
            rdkit.Chem.Fragments.fr_azo(m),
            rdkit.Chem.Fragments.fr_barbitur(m),
            rdkit.Chem.Fragments.fr_benzene(m),
            rdkit.Chem.Fragments.fr_benzodiazepine(m),
            rdkit.Chem.Fragments.fr_bicyclic(m),
            rdkit.Chem.Fragments.fr_diazo(m),
            rdkit.Chem.Fragments.fr_dihydropyridine(m),
            rdkit.Chem.Fragments.fr_epoxide(m),
            rdkit.Chem.Fragments.fr_ester(m),
            rdkit.Chem.Fragments.fr_ether(m),
            rdkit.Chem.Fragments.fr_furan(m),
            rdkit.Chem.Fragments.fr_guanido(m),
            rdkit.Chem.Fragments.fr_halogen(m),
            rdkit.Chem.Fragments.fr_hdrzine(m),
            rdkit.Chem.Fragments.fr_hdrzone(m),
            rdkit.Chem.Fragments.fr_imidazole(m),
            rdkit.Chem.Fragments.fr_imide(m),
            rdkit.Chem.Fragments.fr_isocyan(m),
            rdkit.Chem.Fragments.fr_isothiocyan(m),
            rdkit.Chem.Fragments.fr_ketone(m),
            rdkit.Chem.Fragments.fr_ketone_Topliss(m),
            rdkit.Chem.Fragments.fr_lactam(m),
            rdkit.Chem.Fragments.fr_lactone(m),
            rdkit.Chem.Fragments.fr_methoxy(m),
            rdkit.Chem.Fragments.fr_morpholine(m),
            rdkit.Chem.Fragments.fr_nitrile(m),
            rdkit.Chem.Fragments.fr_nitro(m),
            rdkit.Chem.Fragments.fr_nitro_arom(m),
            rdkit.Chem.Fragments.fr_nitro_arom_nonortho(m),
            rdkit.Chem.Fragments.fr_nitroso(m),
            rdkit.Chem.Fragments.fr_oxazole(m),
            rdkit.Chem.Fragments.fr_oxime(m),
            rdkit.Chem.Fragments.fr_para_hydroxylation(m),
            rdkit.Chem.Fragments.fr_phenol(m),
            rdkit.Chem.Fragments.fr_phenol_noOrthoHbond(m),
            rdkit.Chem.Fragments.fr_phos_acid(m),
            rdkit.Chem.Fragments.fr_phos_ester(m),
            rdkit.Chem.Fragments.fr_piperdine(m),
            rdkit.Chem.Fragments.fr_piperzine(m),
            rdkit.Chem.Fragments.fr_priamide(m),
            rdkit.Chem.Fragments.fr_prisulfonamd(m),
            rdkit.Chem.Fragments.fr_pyridine(m),
            rdkit.Chem.Fragments.fr_quatN(m),
            rdkit.Chem.Fragments.fr_sulfide(m),
            rdkit.Chem.Fragments.fr_sulfonamd(m),
            rdkit.Chem.Fragments.fr_sulfone(m),
            rdkit.Chem.Fragments.fr_term_acetylene(m),
            rdkit.Chem.Fragments.fr_tetrazole(m),
            rdkit.Chem.Fragments.fr_thiazole(m),
            rdkit.Chem.Fragments.fr_thiocyan(m),
            rdkit.Chem.Fragments.fr_thiophene(m),
            rdkit.Chem.Fragments.fr_unbrch_alkane(m),
            rdkit.Chem.Fragments.fr_urea(m)]
    
    f_gr = torch.FloatTensor(f_gr).unsqueeze(dim=0)
    return f_gr