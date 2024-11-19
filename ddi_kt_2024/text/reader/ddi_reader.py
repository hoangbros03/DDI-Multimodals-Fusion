import os
import re
import xmltodict
from utils import id_find

class DDICorpusXMLReader:
    def __init__(self, path):
        self.path = path
        self.train_path = os.path.join(self.path, 'Train')
        self.test_path = os.path.join(self.path, 'Test')
    
    def read_folder(self, 
                    dataset: str):
        '''
        Read XML files of DDI Corpus, return a list of dictionaries,
        each is a document.

        Args:
            dataset (str): 'train' | 'ddiextraction_test' | 'drugner_test'.
        
        Returns:
            drugbank_list (list): a list of dictionaries from drugbank, each is a document.
            medline_list (list): a list of dictionaries from medline, each is a document.
        '''
        if dataset == 'train':
            drugbank_path = os.path.join(self.train_path, 'DrugBank')
            medline_path = os.path.join(self.train_path, 'MedLine')
        elif dataset == 'ddiextraction_test':
            drugbank_path = os.path.join(self.test_path, 'DDIExtraction/DrugBank')
            medline_path = os.path.join(self.test_path, 'DDIExtraction/MedLine')
        elif dataset == 'drugner_test':
            drugbank_path = os.path.join(self.test_path, 'DrugNER/DrugBank')
            medline_path = os.path.join(self.test_path, 'DrugNER/MedLine')
        
        drugbank_list = list()
        medline_list = list()
        
        xml_detect = re.compile('\\.xml$')
            
        for file_path in sorted(os.listdir(drugbank_path)):
            if xml_detect.search(file_path):
                file_abs_path = os.path.join(drugbank_path, file_path)
                with open(file_abs_path, 'r') as file:
                    xml_file = file.read()
                    file_dict = xmltodict.parse(xml_file)
                drugbank_list.append(file_dict)
        
        for file_path in sorted(os.listdir(medline_path)):
            if xml_detect.search(file_path):
                file_abs_path = os.path.join(medline_path, file_path)
                with open(file_abs_path, 'r') as file:
                    xml_file = file.read()
                    file_dict = xmltodict.parse(xml_file)
                medline_list.append(file_dict)
        
        return drugbank_list, medline_list
    
def convert_all_list(ddi_dictionary):
    '''
    Convert every dict to list in ddi_dictionary.
    '''
    for doc in ddi_dictionary:
        doc = doc['document']
        if isinstance(doc['sentence'], dict):
            doc['sentence'] = [doc['sentence']]
        for sentence in doc['sentence']:
            if 'entity' in sentence.keys() and isinstance(sentence['entity'], dict):
                sentence['entity'] = [sentence['entity']]
            if 'pair' in sentence.keys() and isinstance(sentence['pair'], dict):
                sentence['pair'] = [sentence['pair']]
    
    return ddi_dictionary

def get_candidates(ddi_dictionary):
    '''
    Get relation pairs from ddi_dictionary.
    '''
    all_candidates = list()
    for document in ddi_dictionary:
        document = document['document']
        for s in document['sentence']:
            if isinstance(s, dict) and 'pair' in s.keys():
                for pair in s['pair']:
                    candidate = dict()
                    _id = pair['@id']
                    _e1 = pair['@e1']
                    _e2 = pair['@e2']
                    _label = pair['@ddi']
                    candidate['label'] = _label
                    if _label == 'true':
                        try:
                            assert pair['@type'] in ['effect', 'advise', 'mechanism', 'int']
                            candidate['label'] = pair['@type']
                        except:
                            if _id == 'DDI-DrugBank.d236.s29.p0':
                                candidate['label'] = 'int'
                    candidate['id'] = _id
                    candidate['text'] = s['@text']
                    candidate['e1'] = id_find(s['entity'], _e1)
                    candidate['e2'] = id_find(s['entity'], _e2)
                    assert candidate['label'] != 'true'
                    all_candidates.append(candidate)
    return all_candidates