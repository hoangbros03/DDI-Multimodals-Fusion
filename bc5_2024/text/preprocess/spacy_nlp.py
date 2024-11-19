import spacy
from spacy.language import Language
from spacy.attrs import ORTH

class SpacyNLP:
    def __init__(self, spacy_model="en_core_web_lg"):
        self.nlp = spacy.load(spacy_model)
        self.nlp.add_pipe('_custom_sbd', before='parser')
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
        
    @Language.component( '_custom_sbd' )
    def _custom_sbd( doc ):
        for tok in doc:
            tok.is_sent_start = False
        return doc
    
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