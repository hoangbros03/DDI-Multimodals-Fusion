def mapped_smiles_reader(file_path):
    f = open(file_path, 'r')
    all_drugs = f.read().split('\n')[:-1]
    all_drugs = [x.split('|') for x in all_drugs]
    return all_drugs

def find_drug_smiles(mapped_smiles_id, mapped_smiles_name, drug_name):
    drug_name = drug_name.lower()
    for smiles in mapped_smiles_name:
        if drug_name == smiles[1]:
            return smiles[2]
    for smiles in mapped_smiles_id:
        if drug_name == smiles[0]:
            return smiles[1]
    return 'None'

def candidate_smiles(all_candidates, mapped_smiles_id, mapped_smiles_name):
    x = list()
    y = list()
    for c in all_candidates:
        e1 = c['e1']['@text']
        smiles1 = find_drug_smiles(mapped_smiles_id, mapped_smiles_name, e1)
        label = c['label']
        if label == '0':
            label = 0
        elif label == '1':
            label = 1
        x.append([smiles1])
        y.append(label)
    
    return x, y