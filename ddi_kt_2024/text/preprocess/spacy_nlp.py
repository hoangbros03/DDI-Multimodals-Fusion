import spacy
from spacy.language import Language
from spacy.attrs import ORTH

class SpacyNLP:
    def __init__(self, spacy_model="en_core_web_lg"):
        self.nlp = spacy.load(spacy_model)
        self.nlp.add_pipe('_custom_sbd', before='parser')
        
    @Language.component( '_custom_sbd' )
    def _custom_sbd( doc ):
        for tok in doc:
            tok.is_sent_start = False
        return doc