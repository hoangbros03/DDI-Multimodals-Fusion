import re
from tqdm import tqdm

class Cleaner:
    def __init__(self):
        pass

    def clean(self, text):
        text = re.sub(';([^ ])', '; \g<1>', text)
        text = re.sub(':([^ ])', ': \g<1>', text)
        return text
    
    def clean_dictionary(self, ddi_dictionary):
        i = 0
        for doc in tqdm(ddi_dictionary):
            doc = doc['document']
            for sent in doc['sentence']:
                if 'entity' in sent.keys():
                    for entity in sent['entity']:
                        match1 = re.compile(';[^ ]')
                        match2 = re.compile(':[^ ]')
                        offset = entity['@charOffset'].split(';')
                        for i in range(len(offset)):
                            offset[i] = [int(x) for x in offset[i].split('-')]
                            counter = len(re.findall(match1, sent['@text'][0:offset[i][0]+1])) + len(re.findall(match2, sent['@text'][0:offset[i][0]+1]))
                            if counter > 0:
                                offset[i][0] += counter
                                offset[i][1] += counter
                            offset[i] = f"{offset[i][0]}-{offset[i][1]}"
                        if len(offset) > 1:
                            entity['@charOffset'] = ';'.join(offset)
                        else:
                            entity['@charOffset'] = offset[0]
                sent['@text'] = self.clean(sent['@text'])
            i += 1
                        
        return ddi_dictionary