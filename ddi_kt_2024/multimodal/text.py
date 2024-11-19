import os
import torch
from torch import nn
import transformers
from transformers import BertPreTrainedModel, AdamW, BertConfig
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import wandb
import pprint

from ddi_kt_2024.mol.gnn import GNN
from ddi_kt_2024.utils import save_model
from ddi_kt_2024.multimodal.bert import *

class BertForSequenceClassification(BertPreTrainedModel):
    """Desc, formula and text"""
    def __init__(
            self,
            data_type: str = 'ddi_no_negative',
            num_labels: int = 5,
            dropout_rate_other: float = 0.1,
            dropout_rate_bert_output: float = 0.3,
            out_conv_list_dim: int = 128,
            conv_window_size: list[int] = [3, 5, 7],
            max_seq_length: int = 256,
            middle_layer_size: int = 0,
            model_name_or_path: str = 'allenai/scibert_scivocab_uncased',
            pos_emb_dim: int = 10,
            activation: str = "gelu",
            weight_decay: float = 0.01,
            freeze_bert: bool = True,
            config=None,
            apply_mask: bool = False,
            **kwargs
        ):
        super(BertForSequenceClassification, self).__init__(config)
        self.data_type = data_type
        self.num_labels = num_labels
        self.dropout_rate_other = dropout_rate_other
        self.dropout_rate_bert_output = dropout_rate_bert_output
        self.modal_dropout = nn.Dropout(dropout_rate_other)
        self.dropout_bert_output = nn.Dropout(dropout_rate_bert_output)
        self.conv_window_size = conv_window_size
        activations = {'relu':nn.ReLU(), 'elu':nn.ELU(), 'leakyrelu':nn.LeakyReLU(), 'prelu':nn.PReLU(),
                       'relu6':nn.ReLU6, 'rrelu':nn.RReLU(), 'selu':nn.SELU(), 'celu':nn.CELU(), 'gelu':nn.GELU()}
        self.activation = activations[activation]
        self.out_conv_list_dim = out_conv_list_dim
        self.max_seq_length = max_seq_length
        self.pos_emb_dim = pos_emb_dim
        self.middle_layer_size = middle_layer_size
        self.model_name_or_path = model_name_or_path
        self.weight_decay = weight_decay
        self.apply_mask = apply_mask

        # Use cnn
        self.conv_list = nn.ModuleList([nn.Conv1d(768+2*self.pos_emb_dim, out_conv_list_dim, w, padding=(w-1)//2) for w in self.conv_window_size])
        self.pos_emb = nn.Embedding(2*self.max_seq_length, self.pos_emb_dim, padding_idx=0)
        self.norm_text = torch.nn.BatchNorm1d(num_features=len(self.conv_window_size)*self.out_conv_list_dim)

        if self.middle_layer_size == 0:
            self.classifier = nn.Linear(self.out_conv_list_dim*len(self.conv_window_size), config.num_labels)
        else:
            self.middle_classifier = nn.Linear(self.out_conv_list_dim*len(self.conv_window_size), self.middle_layer_size)
            self.classifier = nn.Linear(self.middle_layer_size, config.num_labels)

        self.init_weights()
        self.pos_emb.weight.data.uniform_(-1e-3, 1e-3)
    
        self.bert = transformers.BertModel.from_pretrained(self.model_name_or_path)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.middle_layer_size = self.middle_layer_size
        self._init_weights(self.classifier)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None,
                relative_dist1=None, relative_dist2=None,
                labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        # CNN part
        relative_dist1 *= attention_mask
        relative_dist2 *= attention_mask
        pos_embs1 = self.pos_emb(relative_dist1)
        pos_embs2 = self.pos_emb(relative_dist2)
        conv_input = torch.cat((outputs[0], pos_embs1, pos_embs2), 2)
        conv_outputs = []
        for c in self.conv_list:
            conv_output = self.activation(c(conv_input.transpose(1,2)))
            conv_output, _ = torch.max(conv_output, -1)
            conv_outputs.append(conv_output)
        pooled_output = self.dropout_bert_output(self.norm_text(torch.cat(conv_outputs, 1)))

        if self.middle_layer_size == 0:
            logits = self.classifier(pooled_output)
        else:
            middle_output = self.activation(self.middle_classifier(pooled_output))
            logits = self.classifier(middle_output)
        
        outputs = (logits,) + outputs[2:]

        if self.data_type == 'ddi_no_negative':
            weight = torch.tensor([17029.0/13008, 
                17029.0/826, 
                17029.0/1687, 
                17029.0/1319, 
                17029.0/189]).to('cuda' if torch.cuda.is_available() else 'cpu')
            loss_fct = CrossEntropyLoss(weight=weight)
        else:
            loss_fct = CrossEntropyLoss()
        labels = torch.nn.functional.one_hot(labels, 5).float()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.argmax(dim=-1))                 
        outputs = (loss,) + outputs

        return outputs
    
    def _init_weights(self, module):
        """
        This method initializes weights for different types of layers. The type of layers
        supported are nn.Linear, nn.Embedding and nn.LayerNorm.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class Trainer:
    def __init__(self, 
            num_labels: int = 5,
            dropout_rate_other: float = 0.1,
            dropout_rate_bert_output: float = 0.3,
            out_conv_list_dim: int = 128,
            conv_window_size: list[int] = [3, 5, 7],
            max_seq_length: int = 256,
            pos_emb_dim: int = 10,
            middle_layer_size: int = 256,
            activation: str = "gelu",
            parameter_averaging: bool = False,
            epsilon: float = 1e-8,
            lr: float = 1e-4,
            optimizer: str = 'rmsprop',
            weight_decay: float = 0.01,
            model_name_or_path: str = "allenai/scibert_scivocab_uncased",
            wandb_available: bool = False,
            freeze_bert: bool = False,
            log: bool = True,
            data_type: str = 'ddi_no_negative',
            device: str = 'cuda',
            apply_mask: bool = False,
            **kwargs):
        self.parameter_averaging = parameter_averaging
        self.lr = lr
        self.wandb_available = wandb_available
        self.freeze_bert = freeze_bert
        self.dropout_rate_other = dropout_rate_other
        self.dropout_rate_bert_output = dropout_rate_bert_output
        self.config = BertConfig.from_pretrained(model_name_or_path)
        self.train_loss = list()
        self.train_micro_f1 = list()
        self.val_loss = list()
        self.val_micro_f1 = list()
        self.val_macro_f1 = list()
        self.val_precision = list()
        self.val_recall = list()
        self.val_f_mechanism = list()
        self.val_f_effect = list()
        self.val_f_advise = list()
        self.val_f_int = list()
        self.log = log
        self.device = device

        self.model = BertForSequenceClassification(
            num_labels=num_labels,
            dropout_rate_other=dropout_rate_other,
            dropout_rate_bert_output=dropout_rate_bert_output,
            out_conv_list_dim=out_conv_list_dim,
            conv_window_size=conv_window_size,
            max_seq_length=max_seq_length,
            middle_layer_size=middle_layer_size,
            model_name_or_path=model_name_or_path,
            pos_emb_dim=pos_emb_dim,
            activation=activation,
            weight_decay=weight_decay,
            freeze_bert=True,
            config=self.config,
            data_type=data_type,
            apply_mask=apply_mask,
            **kwargs
        )
        
        self.weight_decay = weight_decay
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if optimizer == 'adamw':
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=epsilon)
        elif optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(optimizer_grouped_parameters, lr=lr, eps=epsilon)
            
    def train(self, training_loader, validation_loader, num_epochs,
                    filtered_lst_index=None, full_label=None):
        """ Train the model """
        print("""
        - Modal 1: Desc 
        - Modal 2: Formula
        Separated as 2 entities as normal.
        """)
        train_loss = 0.0
        max_val_micro_f1 = 0.0
        nb_train_step = 0
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=num_epochs
        )

        self.model.zero_grad()
        self.model.to(self.device)

        for epoch in range(num_epochs):
            print(f"================Epoch {epoch}================")
            for step, (batch) in tqdm(enumerate(training_loader)):
                nb_train_step += 1
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'relative_dist1': batch[3],
                          'relative_dist2': batch[4],
                          'labels':         batch[5]}
                    
                outputs = self.model(**inputs)
                loss = outputs[0]
                loss.backward()

                train_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

                self.optimizer.step()
                self.model.zero_grad()

            self.scheduler.step()

            print(f"LR: {self.scheduler.get_last_lr()}")

            train_loss = train_loss / nb_train_step
            self.train_loss.append(train_loss)
            
            results, preds = self.evaluate(validation_loader,
                                    filtered_lst_index,
                                    full_label,
                                    option='val')
            
            # Save model checkpoint
            if max_val_micro_f1 < results['microF']:
                if not os.path.exists("checkpoints"):
                    os.makedirs("checkpoints")
                save_model(
                        output_path=f"checkpoints", 
                        file_name=f"epoch{epoch}val_micro_f1{results['microF']}.pt",
                        config=None, 
                        model=self.model, 
                        pred_logit=preds, 
                        wandb_available=True
                    )
                max_val_micro_f1 = results['microF']
                print(f"Checkpoint saved at:\n  Epoch = {epoch}\n  Micro F1 = {max_val_micro_f1}")

        if self.wandb_available:
            wandb.finish()

    def ddie_compute_metrics(self, preds, labels, every_type=True):
        label_list = ('Advise', 'Effect', 'Mechanism', 'Int.')
        p, r, f, s = precision_recall_fscore_support(y_pred=preds, y_true=labels, labels=[1,2,3,4], average='micro')
        result = {
            "Precision": p,
            "Recall": r,
            "microF": f
        }
        p, r, f, s = precision_recall_fscore_support(y_pred=preds, y_true=labels, labels=[1,2,3,4], average='macro')
        result['macroF'] = f
        if every_type:
            evaluation = precision_recall_fscore_support(y_pred=preds, y_true=labels, labels=[1,2,3,4], average=None)
            for i, label_type in enumerate(label_list):
                result[label_type + ' Precision'] = evaluation[0][i]
                result[label_type + ' Recall'] = evaluation[1][i]
                result[label_type + ' F'] = evaluation[2][i]
        return result
    
    def convert_prediction_to_full_prediction(self, 
                                              prediction, 
                                              filtered_lst_index,
                                              full_label):
        full_predictions = list()
        full_length = len(full_label)
        tmp_prediction = 0
        tmp_full = 0
        for i in range(full_length):
            if i in filtered_lst_index:
                full_predictions.append(prediction[tmp_prediction])
                tmp_prediction += 1
            else:
                full_predictions.append(0)
                tmp_full += 1

        return np.array(full_predictions)
    
    def evaluate(self, 
                 validation_loader, 
                 filtered_lst_index=None,
                 full_label=None,
                 option='val'):
        # Eval!
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for (batch) in validation_loader:
            self.model.eval()
            
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'relative_dist1': batch[3],
                          'relative_dist2': batch[4],
                          'labels':         batch[5]}

                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
            
        eval_loss = eval_loss / nb_eval_steps

        preds = np.argmax(preds, axis=1)

        full_predictions = self.convert_prediction_to_full_prediction(preds,
                                                                      filtered_lst_index,
                                                                      full_label)
        
        result = self.ddie_compute_metrics(full_predictions, full_label)

        if option == 'train':
            self.train_loss.append(eval_loss)
            self.train_micro_f1.append(result['microF'])
        elif option == 'val':
            self.val_loss.append(eval_loss)
            self.val_micro_f1.append(result['microF'])
            self.val_macro_f1.append(result['macroF'])
            self.val_precision.append(result['Precision'])
            self.val_recall.append(result['Recall'])
            self.val_f_mechanism.append(result['Mechanism F'])
            self.val_f_effect.append(result['Effect F'])
            self.val_f_advise.append(result['Advise F'])
            self.val_f_int.append(result['Int. F'])

        pprint.pprint(result)
            
        if self.log == True:
            wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                            y_true=preds, preds=out_label_ids,
                            class_names=['false', 'advise', 'effect', 'mechanism', 'int']),
                        "train_loss": self.train_loss[-1],
                        "val_loss": self.val_loss[-1],
                        "val_precision": self.val_precision[-1], 
                        "val_recall": self.val_recall[-1], 
                        "val_micro_f1": self.val_micro_f1[-1],
                        "val_macro_f1": self.val_macro_f1[-1],
                        "val_f_advise": self.val_f_advise[-1],
                        "val_f_effect": self.val_f_effect[-1],
                        "val_f_mechanism": self.val_f_mechanism[-1],
                        "val_f_int": self.val_f_int[-1]})
        
        return result, preds