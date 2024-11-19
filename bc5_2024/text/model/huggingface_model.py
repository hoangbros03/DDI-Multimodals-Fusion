from transformers import AutoTokenizer, BertModel

def get_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    return tokenizer, model