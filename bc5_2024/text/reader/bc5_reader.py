import pandas as pd
from collections import defaultdict
import re
import spacy
from spacy.attrs import ORTH


class Reader:
    def __init__(self, file_name):
        self.file_name = file_name

    def read(self, **kwargs):
        """
        return raw data from input file
        :param kwargs:
        :return:
        """
        pass


class BioCreativeReader(Reader):
    def __init__(self, file_name):
        super().__init__(file_name)

        with open(file_name, 'r') as f:
            self.lines = f.readlines()

    def read(self):
        """
        :return: dict of abstract's: {<id>: {'t': <string>, 'a': <string>}}
        """
        regex = re.compile(r'^([\d]+)\|([at])\|(.+)$', re.U | re.I)
        abstracts = defaultdict(dict)

        for line in self.lines:
            matched = regex.match(line)
            if matched:
                data = matched.groups()
                abstracts[data[0]][data[1]] = data[2]

        return abstracts

    def read_entity(self):
        """
        :return: dict of entity's: {<id>: [(pmid, start, end, content, type, id)]}
        """
        regex = re.compile(r'^(\d+)\t(\d+)\t(\d+)\t([^\t]+)\t(\S+)\t(\S+)', re.U | re.I)

        ret = defaultdict(list)

        for line in self.lines:
            matched = regex.search(line)
            if matched:
                data = matched.groups()
                ret[data[0]].append(tuple([data[0], int(data[1]), int(data[2]), data[3], data[4], data[5]]))

        return ret

    def read_relation(self):
        """
        :return: dict of relation's: {<id>: [(pmid, type, chem_id, dis_id)]}
        """
        regex = re.compile(r'^([\d]+)\t(CID)\t([\S]+)\t([\S]+)$', re.U | re.I)
        ret = defaultdict(list)

        for line in self.lines:
            matched = regex.match(line)
            if matched:
                data = matched.groups()
                ret[data[0]].append(data)

        return ret

class IntraSentenceDataCreator:
    def __init__(self):
        pass

    def get_data(self, abstract, entity, relation, preprocesser):
        from tqdm import tqdm

        abstract_out = []
        entity_out = []
        relation_out = []
        pair_out = []
        sentences_out = []

        for index in tqdm(abstract):

            sent_abstract_split = []
            sent_abstract = []
            sent_entity = []
            sent_relation = []
            sent_pair = []

            text = abstract[index]['t'] + " " + abstract[index]['a']

            # Add abstract of splitted sentences
            sent_abstract_split = preprocesser.sentTokenizer(text)
            sentences_out.append(sent_abstract_split)

            for i in range(len(sent_abstract_split)):
                tmp = (index + "_" + str(i), sent_abstract_split[i])
                sent_abstract.append(tmp)

            # Add entities of splitted sentences
            sent_length = []
            length_counter = 0
            sent_pos = 0

            for i in sent_abstract:
                if sent_pos == 0:
                    length_counter += len(i[1])
                    sent_length.append(length_counter)
                    sent_pos += 1
                else:
                    length_counter += len(i[1]) + 1
                    sent_length.append(length_counter)
                    sent_pos += 1

            for i in entity[index]:
                sent_pos = 0
                place = i[1]
                while place > sent_length[sent_pos]:
                    sent_pos += 1
                tmp = list(i)
                tmp[0] = index + "_" + str(sent_pos)
                if sent_pos > 0:
                    tmp[1] -= sent_length[sent_pos - 1] + 1
                    tmp[2] -= sent_length[sent_pos - 1] + 1
                i = tuple(tmp)
                sent_entity.append(i)

            # Add relations of splitted sentences
            for r in relation[index]:
                for e1 in sent_entity:
                    if e1[5] == r[2]:
                        for e2 in sent_entity:
                            if (e2[5] == r[3]) and (e2[0] == e1[0]):
                                tmp = list(r)
                                tmp[0] = e1[0]
                                tmp = tuple(tmp)
                                sent_relation.append(tmp)

            # Add chemical-disease pairs and labels to document data
            for e1 in sent_entity:
                if e1[4] == 'Chemical':
                    for e2 in sent_entity:
                        if e1[0] == e2[0] and e2[4] == 'Disease':
                            tmp = (e1[0], e1[1], e1[2], e1[3], e1[4], e1[5],
                                e2[1], e2[2], e2[3], e2[4], e2[5], 0)
                            sent_pair.append(tmp)

            for i in range(len(sent_pair)):
                for r in sent_relation:
                    if sent_pair[i][5] == r[2] and sent_pair[i][10] == r[3]:
                        sent_pair[i] = list(sent_pair[i])
                        sent_pair[i][11] = 1
                        sent_pair[i] = tuple(sent_pair[i])

            # Add data of document to dataset
            abstract_out.append(sent_abstract)
            entity_out.append(sent_entity)
            relation_out.append(sent_relation)
            pair_out.append(sent_pair)

        return abstract_out, entity_out, relation_out, pair_out

    def find_sent_given_id(self, intra_abstract, sent_id):
        '''
        Find sentence given sentence ID
        '''
        for text in intra_abstract:
            for sent in text:
                if sent[0] == sent_id:
                    return sent[1]

    def add_abs_id_col(self, df):
        '''
        Add abstract ID to each row in dataframe
        '''
        abs = list()
        for i in range(len(df)):
            abs_id = df.iloc[i]['sent_id'].split('_')[0]
            abs.append(abs_id)

        df.insert(loc=0, column='abs_id', value=abs)
        return df

    def convert_to_df(self, intra_abstract, intra_pair):
        '''
        Given information about intra-sentence entities and relations,
        return dataframe to display intra-sentence relation labels
        '''

        df = pd.DataFrame(columns=['sent_id', 'text', 'ent1_start', 'ent1_end',
                                   'ent1_name', 'ent1_type', 'ent1_id', 'ent2_start',
                                   'ent2_end', 'ent2_name', 'ent2_type', 'ent2_id', 'label'])

        for text in intra_pair:
            for sent in text:
                sent_id = sent[0]
                sent_tmp = self.find_sent_given_id(intra_abstract, sent_id)
                row = list(sent)
                row = [row[0]] + [sent_tmp] + row[1:]
                df.loc[len(df.index)] = row

        return self.add_abs_id_col(df)

    def get_unique_tuple(self, df):
        '''
        Return the unique tuples of (sent_id, ent1_id, ent2_id) in the df
        Ambiguous entities are splitted into multiple tuples
        '''
        all_tuple = list()
        for i in range(len(df)):
            ent1_id_lst = df.iloc[i]['ent1_id'].split('|')
            ent2_id_lst = df.iloc[i]['ent2_id'].split('|')
            for x in ent1_id_lst:
                for y in ent2_id_lst:
                    if x != '-1' and y != '-1':
                        all_tuple.append((df.iloc[i]['abs_id'], x, y))
        return sorted(set(all_tuple), key=all_tuple.index)

    def get_all_tuple(self, df):
        '''
        Get all tuples of (sent_id, ent1_id, ent2_id)
        Each row in df is a list of tuples in returned list
        '''
        all_tuple = list()
        for i in range(len(df)):
            ent1_id_lst = df.iloc[i]['ent1_id'].split('|')
            ent2_id_lst = df.iloc[i]['ent2_id'].split('|')
            tmp = list()
            for x in ent1_id_lst:
                for y in ent2_id_lst:
                    tmp.append((df.iloc[i]['abs_id'], x, y))
            all_tuple.append(tmp)

        return all_tuple

    def get_candidate(self, df):
        unique_tuple = self.get_unique_tuple(df)
        all_tuple = self.get_all_tuple(df)
        dct = {}
        for tpl in unique_tuple:
            tmp = list()
            for i in range(len(all_tuple)):
                if tpl in all_tuple[i]:
                    tmp.append(i)
            dct[tpl] = tmp

        return dct

    def plot_frequency(self, lookup):
        from collections import Counter
        import matplotlib.pyplot as plt
        freq_lst = list()
        for k, v in lookup.items():
            freq_lst.append(len(v))
        counts = Counter(freq_lst)
        integers = list(counts.keys())
        frequencies = list(counts.values())

        # Plot the histogram
        plt.figure(figsize=(12, 9))
        plt.bar(integers, frequencies)
        plt.xlabel("Number")
        plt.ylabel("Frequency")
        plt.title("Number of intra-sentence candidates per chemical-disease pair")
        plt.show()

class SpacyFeatures:
    def __init__(self, spacy_model="en_core_web_lg"):
        self.nlp = spacy.load(spacy_model)
        self.nlp.tokenizer.add_special_case(u'+/-', [{ORTH: u'+/-'}])
        self.nlp.tokenizer.add_special_case("mg.", [{ORTH: "mg."}])
        self.nlp.tokenizer.add_special_case("mg/kg", [{ORTH: "mg/kg"}])
        self.nlp.tokenizer.add_special_case("Gm.", [{ORTH: "Gm."}])
        self.nlp.tokenizer.add_special_case("i.c.", [{ORTH: "i.c."}])
        self.nlp.tokenizer.add_special_case("i.p.", [{ORTH: "i.p."}])
        self.nlp.tokenizer.add_special_case("s.c.", [{ORTH: "s.c."}])
        self.nlp.tokenizer.add_special_case("p.o.", [{ORTH: "p.o."}])
        self.nlp.tokenizer.add_special_case("i.c.v.", [{ORTH: "i.c.v."}])
        self.nlp.tokenizer.add_special_case("e.g.", [{ORTH: "e.g."}])
        self.nlp.tokenizer.add_special_case("i.v.", [{ORTH: "i.v."}])
        self.nlp.tokenizer.add_special_case("t.d.s.", [{ORTH: "t.d.s."}])
        self.nlp.tokenizer.add_special_case("t.i.d.", [{ORTH: "t.i.d."}])
        self.nlp.tokenizer.add_special_case("b.i.d.", [{ORTH: "b.i.d."}])
        self.nlp.tokenizer.add_special_case("i.m.", [{ORTH: "i.m."}])
        self.nlp.tokenizer.add_special_case("i.e.", [{ORTH: "i.e."}])
        self.nlp.tokenizer.add_special_case("medications.", [{ORTH: "medications."}])
        self.nlp.tokenizer.add_special_case("mEq.", [{ORTH: "mEq."}])
        self.nlp.tokenizer.add_special_case("a.m.", [{ORTH: "a.m."}])
        self.nlp.tokenizer.add_special_case("p.m.", [{ORTH: "p.m."}])
        self.nlp.tokenizer.add_special_case("M.S.", [{ORTH: "M.S."}])
        self.nlp.tokenizer.add_special_case("ng.", [{ORTH: "ng."}])
        self.nlp.tokenizer.add_special_case("ml.", [{ORTH: "ml."}])
        self.nlp.tokenizer.add_special_case(u'sgk1(+/+)', [{ORTH: u'sgk1(+/+)'}])
        
    def wordTokenizer(self, text):
        doc = self.nlp(text)
        return [(token.text, token.idx, token.i) for token in doc]

    def sentTokenizer(self, sent):
        doc = self.nlp(sent)
        return [sent.text for sent in doc.sents]
    
    def tag(self, text):
        doc = self.nlp(text)
        return [token.tag_ for token in doc]

    def POS(self, text):
        doc = self.nlp(text)
        return [token.pos_ for token in doc]

    def dependencyTagger(self, text):
        doc = self.nlp(text)
        return [token.dep_ for token in doc]

    def IOBTagger(self, text):
        doc = self.nlp(text)
        return [token.ent_iob_ for token in doc]

def process_bc5(path):
    spacy_features = SpacyFeatures('en_core_web_lg')
    bc5cdr_reader = BioCreativeReader(path)
    abstract = bc5cdr_reader.read()
    entity = bc5cdr_reader.read_entity()
    relation = bc5cdr_reader.read_relation()
    intra_sentence_data_creator = IntraSentenceDataCreator()
    a, e, r, p = intra_sentence_data_creator.get_data(abstract, entity, relation, spacy_features)
    df = intra_sentence_data_creator.convert_to_df(a, p)
    return df

def convert_bc5_to_dict(df):
    dict_list = list()
    for i in range(len(df)):
        dct = {'label': df.iloc[i]['label'],
               'id': df.iloc[i]['sent_id'],
               'text': df.iloc[i]['text'],
               'e1': {"@id": df.iloc[i]['ent1_id'],
                     '@charOffset': '-'.join([str(df.iloc[i]['ent1_start']), str(df.iloc[i]['ent1_end'])]),
                     '@type': 'Chemical',
                     '@text': df.iloc[i]['ent1_name']},
               'e2': {"@id": df.iloc[i]['ent2_id'],
                     '@charOffset': '-'.join([str(df.iloc[i]['ent2_start']), str(df.iloc[i]['ent2_end'])]),
                     '@type': 'Disease',
                     '@text': df.iloc[i]['ent2_name']}}
        dict_list.append(dct)
        
    return dict_list