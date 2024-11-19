import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BertModel

from bc5_2024.utils import offset_to_idx, idx_to_offset

class PathProcesser:
    def __init__(self, spacy_nlp, lookup_word, lookup_dep, lookup_tag, lookup_direction, bert_model_path, device='cpu'):
        self.spacy_nlp = spacy_nlp
        self.lookup_word = lookup_word
        self.lookup_dep = lookup_dep
        self.lookup_tag = lookup_tag
        self.lookup_direction = lookup_direction
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
        self.bert_model = BertModel.from_pretrained(bert_model_path)
        self.bert_model.to(device)
        self.device = device
        
    def get_position_embedding_given_ent(self, 
                                         ent_start: int, 
                                         ent_end: int, 
                                         text_length: int):
        '''
        Given entity index, get position embedding of sentence
        '''
        lst = []
        count_bef = ent_start
        count_in = ent_end - ent_start
        count_aft = text_length - ent_end - 1

        for i in range(count_bef, 0, -1):
            lst.append(-i)
        
        for i in range(count_in + 1):
            lst.append(0)

        for i in range(1, count_aft + 1):
            lst.append(i)
        return lst
    
    def build_position_embedding(self, text, offset):
        doc = self.spacy_nlp.nlp(text)
        text_length = len(doc)
        start_idx, end_idx = offset_to_idx(text, offset, self.spacy_nlp.nlp)
        pos_ent = self.get_position_embedding_given_ent(start_idx, end_idx, text_length)
        zero_ent = [0] * text_length
        for i in range(start_idx, end_idx + 1):
            zero_ent[i] = 1
        return [pos_ent, zero_ent]
    
    def create_mapping(self, candidate, sdp):
        text = candidate['text']
        doc = self.spacy_nlp.nlp(text)
        word_index = list()
        mapped_sdp = list()
        
        for tok in doc:
            pos = tok.i
            tag_key = tok.tag_
            word_key = tok.text
            try:
                word_index.append([self.lookup_word[word_key], 
                                   self.lookup_tag[tag_key]])
            except: 
                if word_key not in self.lookup_word.keys():
                    word_index.append([12367, self.lookup_tag[tag_key]])
        
        position_embedding_ent1 = self.build_position_embedding(text, candidate['e1']['@charOffset'])
        position_embedding_ent2 = self.build_position_embedding(text, candidate['e2']['@charOffset'])
        
        for edge in sdp:
            word1_idx = edge[0]
            word2_idx = edge[2]
            word1_keys = word_index[word1_idx] + [position_embedding_ent1[0][word1_idx], position_embedding_ent2[0][word1_idx], position_embedding_ent1[1][word1_idx], position_embedding_ent2[1][word1_idx]]
            word2_keys = word_index[word2_idx] + [position_embedding_ent1[0][word2_idx], position_embedding_ent2[0][word2_idx], position_embedding_ent1[1][word2_idx], position_embedding_ent2[1][word2_idx]]
            edge_keys = [self.lookup_direction[edge[1][0]], self.lookup_dep[edge[1][1]]]
            mapped_sdp.append(word1_keys + edge_keys + word2_keys)
        
        return torch.tensor(mapped_sdp)
    
    def return_bert_position(self, left_offset, right_offset, offset_mapping):
        pos_list = list()
        for i in range(len(offset_mapping[0][1:-1])):
            if left_offset < offset_mapping[0][1:-1][i][0]:
                j = i - 1
                while right_offset >= offset_mapping[0][1:-1][j][1]:
                    pos_list.append(j)
                    if j + 1 < len(offset_mapping[0][1:-1]):
                        j += 1
                    else: 
                        return(pos_list[0], pos_list[-1])
                if len(pos_list) == 0:
                    if left_offset > offset_mapping[0][1:-1][-1][0]:
                        return (len(offset_mapping[0][1:-1]) -1, len(offset_mapping[0][1:-1]) -1)
                    for i in range(len(offset_mapping[0][1:-1])):
                        if left_offset >= offset_mapping[0][1:-1][i][0]:
                            pos_list.append(i)
                            for j in range(i, len(offset_mapping[0][1:-1])):
                                if right_offset <= offset_mapping[0][1:-1][j][0]:
                                    return (i,j)
                            
                            return (i, len(offset_mapping[0][1:-1]) - 1)

                return (pos_list[0], pos_list[-1])
        
        return (len(offset_mapping[0][1:-1]) - 1, len(offset_mapping[0][1:-1]) - 1)
        
    def create_mapping_with_bert(self, candidate, sdp, text, offset_mapping, temp_result):
        if sdp == []:
            return torch.zeros([1, 1550]).to(self.device)
        doc = self.spacy_nlp.nlp(text)
        word_index = list()
        mapped_sdp = list()
        
        for tok in doc:
            pos = tok.i
            tag_key = tok.tag_
            word_key = tok.text
            try:
                word_index.append([self.lookup_word[word_key], 
                                   self.lookup_tag[tag_key]])
            except: 
                if word_key not in self.lookup_word.keys():
                    word_index.append([12367, self.lookup_tag[tag_key]])

        position_embedding_ent1 = self.build_position_embedding(text, candidate['e1']['@charOffset'])
        position_embedding_ent2 = self.build_position_embedding(text, candidate['e2']['@charOffset'])
        
#         sentence_tokenize = self.tokenizer.convert_ids_to_tokens(encoding[0])[1:-1]
#         for i in range(len(sentence_tokenize)):
#             print(f"{i}: {sentence_tokenize[i]}")
            
#         print(sdp)
        
        for edge in sdp:
            word1_idx = edge[0]
            word2_idx = edge[2]
            
            # Get BERT embedding
            word1_offset = idx_to_offset(text, word1_idx, self.spacy_nlp.nlp)
            word2_offset = idx_to_offset(text, word2_idx, self.spacy_nlp.nlp)
        
            word1_bert_pos = self.return_bert_position(word1_offset[0], word1_offset[1], offset_mapping)
            word2_bert_pos = self.return_bert_position(word2_offset[0], word2_offset[1], offset_mapping)

            word1_bert_embedding = torch.mean(temp_result[:, word1_bert_pos[0]:word1_bert_pos[1]+1, :], dim=1)
            word2_bert_embedding = torch.mean(temp_result[:, word2_bert_pos[0]:word2_bert_pos[1]+1, :], dim=1)
#             print(word1_bert_pos, word2_bert_pos)

            word1_keys = torch.tensor(word_index[word1_idx] + [position_embedding_ent1[0][word1_idx], position_embedding_ent2[0][word1_idx], position_embedding_ent1[1][word1_idx], position_embedding_ent2[1][word1_idx]]).unsqueeze(dim=0).to(self.device)
            word2_keys = torch.tensor(word_index[word2_idx] + [position_embedding_ent1[0][word2_idx], position_embedding_ent2[0][word2_idx], position_embedding_ent1[1][word2_idx], position_embedding_ent2[1][word2_idx]]).unsqueeze(dim=0).to(self.device)
            edge_keys = torch.tensor([self.lookup_direction[edge[1][0]], self.lookup_dep[edge[1][1]]]).unsqueeze(dim=0).to(self.device)
            mapped_sdp.append(torch.cat((word1_keys, edge_keys, word2_keys, word1_bert_embedding, word2_bert_embedding), dim=1))

        return torch.cat(mapped_sdp)
    
    def create_mapping_all(self, all_candidates, all_sdp, option='bert'):
        all_mapped_sdp = list()
        for i in tqdm(range(len(all_candidates))):
            if option == 'normal':
                mapped_sdp = self.create_mapping(all_candidates[i], all_sdp[i]).unsqueeze(dim=0)
            elif option == 'bert':
                if i == 0 or all_candidates[i]['text'] != all_candidates[i-1]['text']:
                    text = all_candidates[i]['text']
                    output = self.tokenizer([text], return_offsets_mapping=True)
                    encoding = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
                    temp_result = self.bert_model(encoding).last_hidden_state.detach()[:,1:-1,:]
                
                mapped_sdp = self.create_mapping_with_bert(all_candidates[i], all_sdp[i], text, output['offset_mapping'], temp_result).unsqueeze(dim=0)
            all_mapped_sdp.append(mapped_sdp)
            
        return all_mapped_sdp