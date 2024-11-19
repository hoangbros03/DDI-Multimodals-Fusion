import pandas as pd
import re

def mapped_property_reader(file_path):
    df = pd.read_csv(file_path)
    return df

def get_property_dict(df, property_name):
    query_dict = dict()
    for i in range(len(df)):
        query_dict[df.iloc[i]['name'].lower()] = df.iloc[i][property_name]
        if type(query_dict[df.iloc[i]['name'].lower()]) != str:
            query_dict[df.iloc[i]['name'].lower()] = 'None'
    return query_dict

def find_drug_property(drug_name, query_dict):
    return query_dict[drug_name.lower()]

def candidate_property(all_candidates, query_dict):
    x = list()
    y = list()
    for c in all_candidates:
        e1 = c['e1']['@text']
        e2 = c['e2']['@text']
        smiles1 = find_drug_property(e1, query_dict)
        smiles2 = find_drug_property(e2, query_dict)
        label = c['label']
        if label == 'false':
            label = 0
        elif label == 'advise':
            label = 1
        elif label == 'effect':
            label = 2
        elif label == 'mechanism':
            label = 3
        elif label == 'int':
            label = 4
        x.append([smiles1, smiles2])
        y.append(label)
    
    return x, y

def split_formula(string, option='1'):
    if option == '0':
        return [c for c in string]
    elif option == '1':
        char_list = [c for c in string]
        char_list_new = list()
        
        tmp = ''
        for i in range(len(char_list)):
            if re.match('[0-9]', char_list[i]) and re.match('[0-9]', char_list[i-1]):
                tmp += char_list[i]
            elif re.match('[0-9]', char_list[i]):
                char_list_new.append(tmp)
                tmp = char_list[i]
            elif re.match('[a-z]', char_list[i]):
                tmp += char_list[i]
            elif re.match('[A-Z]', char_list[i]):
                char_list_new.append(tmp)
                tmp = char_list[i]
            
        char_list_new.append(tmp)
        char_list_new = char_list_new[1:]
        
        char_list_return = list()
        for i in range(len(char_list_new)):
            if i == len(char_list_new) - 1:
                if re.match('[a-zA-Z]+', char_list_new[i]):
                    char_list_return.append(char_list_new[i])
                    char_list_return.append('1')
                else:
                    char_list_return.append(char_list_new[i])
            elif re.match('[a-zA-Z]+', char_list_new[i]) and re.match('[a-zA-Z]+', char_list_new[i+1]):
                char_list_return.append(char_list_new[i])
                char_list_return.append('1')
            else:
                char_list_return.append(char_list_new[i])
                
        return char_list_return
    elif option == '2':
        map_abrv_dict = {
            'Ag': 'Silver',
            'Al': 'Aluminum',
            'As': 'Arsenic',
            'Au': 'Gold',
            'B': 'Boron',
            'Ba': 'Barium',
            'Bi': 'Bismuth',
            'Br': 'Bromine',
            'C': 'Carbon',
            'Ca': 'Calcium',
            'Cl': 'Chlorine',
            'Co': 'Cobalt',
            'Cu': 'Copper',
            'F': 'Fluorine',
            'Fe': 'Iron',
            'H': 'Hydrogen',
            'Hg': 'Mercury',
            'I': 'Iodine',
            'K': 'Potassium',
            'Li': 'Lithium',
            'Mg': 'Magnesium',
            'N': 'Nitrogen',
            'Na': 'Sodium',
            'O': 'Oxygen',
            'P': 'Phosphorus',
            'Pt': 'Platinum',
            'S': 'Sulfur',
            'Se': 'Selenium',
            'Si': 'Silicon',
            'Tc': 'Technetium',
            'Ti': 'Titanium',
            'Zn': 'Zinc'
        }

        char_list = [c for c in string]
        char_list_new = list()
        
        tmp = ''
        for i in range(len(char_list)):
            if re.match('[0-9]', char_list[i]) and re.match('[0-9]', char_list[i-1]):
                tmp += char_list[i]
            elif re.match('[0-9]', char_list[i]):
                char_list_new.append(tmp)
                tmp = char_list[i]
            elif re.match('[a-z]', char_list[i]):
                tmp += char_list[i]
            elif re.match('[A-Z]', char_list[i]):
                char_list_new.append(tmp)
                tmp = char_list[i]
            
        char_list_new.append(tmp)
        char_list_new = char_list_new[1:]
        
        char_list_return = list()
        for i in range(len(char_list_new)):
            if i == len(char_list_new) - 1:
                if re.match('[a-zA-Z]+', char_list_new[i]):
                    char_list_return.append(char_list_new[i])
                    char_list_return.append('1')
                else:
                    char_list_return.append(char_list_new[i])
            elif re.match('[a-zA-Z]+', char_list_new[i]) and re.match('[a-zA-Z]+', char_list_new[i+1]):
                char_list_return.append(char_list_new[i])
                char_list_return.append('1')
            else:
                char_list_return.append(char_list_new[i])

        for i in range(len(char_list_return)):
            if char_list_return[i] in map_abrv_dict.keys():
                char_list_return[i] = map_abrv_dict[char_list_return[i]]
                
        return char_list_return