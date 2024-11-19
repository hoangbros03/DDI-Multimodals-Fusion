import torch
from torch import nn
from transformers import AutoTokenizer, BertModel, RobertaForMaskedLM

class NHotClassification(nn.Module):
    def __init__(self,
                 dropout_rate: float = 0.1,
                 input_dims: int = 85,
                 layer1_dims: int = 512,
                 output_dims: int = 128,
                 **kwargs):
        super(NHotClassification, self).__init__()
        self.input_dims = input_dims
        self.layer1_dims = layer1_dims
        self.output_dims = output_dims
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(input_dims, layer1_dims)
        self.fc2 = nn.Linear(layer1_dims, output_dims)
        self.activation = nn.GELU()

    def forward(self, x):
        x = x.float()
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.squeeze()
        return x
    
class BertForFormulaClassification(nn.Module):
    def __init__(self, 
                 model_name: str='allenai/scibert_scivocab_uncased',
                 dropout_rate: float=0.1,
                 **kwargs):
        super(BertForFormulaClassification, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_dims = 768
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
        
        outputs = self.dropout(outputs[0])
        
        bincount = [torch.bincount(i)[1] for i in attention_mask]
        
        mask = list()
        for i in bincount:
            if i == 2:
                mask.append(0)
            else:
                mask.append(1)
                
        mask = torch.FloatTensor(mask).unsqueeze(dim=1).to(outputs[1].device)
        pooled_output = outputs

        return pooled_output, mask

class BertForSmilesClassification(nn.Module):
    def __init__(self,
                 model_name: str='seyonec/ChemBERTa-zinc-base-v1',
                 bert_to_embedding_option: str='mean',
                 model_output_dims: int=767,
                 output_dims: int=128,
                 **kwargs):
        super(BertForSmilesClassification, self).__init__()
        self.output_dims = output_dims
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = RobertaForMaskedLM.from_pretrained(model_name)
        self.readout = nn.Linear(model_output_dims, output_dims)
        self.bert_to_embedding_option = bert_to_embedding_option
        
    def forward(self, input_ids=None, attention_mask=None):

        outputs = self.model(input_ids,
                             attention_mask=attention_mask)
        
        pooled_output = outputs.logits

        if self.bert_to_embedding_option == 'mean':
            pooled_output = pooled_output.mean(dim=1)
        elif self.bert_to_embedding_option == 'cls':
            pooled_output = pooled_output[:, 0, :]

        pooled_output = self.readout(pooled_output)

        return pooled_output