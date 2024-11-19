import os
import torch
from torch import nn
from transformers import BertPreTrainedModel, AdamW, BertConfig
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import wandb

from ddi_kt_2024.multimodal.bert import *
from ddi_kt_2024.mol.gnn import GNN
from ddi_kt_2024.utils import save_model

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self,
                num_labels,
                dropout_rate,
                middle_layer_size,
                activation="gelu",
                weight_decay=0.0,
                data_type="other",
                config=None,
                apply_mask=False,
                **kwargs
                ):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        activations = {'relu':nn.ReLU(), 'elu':nn.ELU(), 'leakyrelu':nn.LeakyReLU(), 'prelu':nn.PReLU(),
                       'relu6':nn.ReLU6, 'rrelu':nn.RReLU(), 'selu':nn.SELU(), 'celu':nn.CELU(), 'gelu':nn.GELU()}
        self.activation = activations[activation]
        self.middle_layer_size = middle_layer_size
        self.weight_decay = weight_decay
        self.data_type = data_type
        self.apply_mask = apply_mask

        self.gnn = GNN(**kwargs)
        self.norm_graph = torch.nn.BatchNorm1d(self.gnn.hidden_channels)
        print(f"Using Graph. Output dims = {self.gnn.hidden_channels}")
        if self.middle_layer_size == 0:
            self.classifier = nn.Linear(2*self.gnn.hidden_channels, 5)
        else:
            self.middle_classifier = nn.Linear(2*self.gnn.hidden_channels, self.middle_layer_size)
            self.classifier = nn.Linear(self.middle_layer_size, 5)

    def forward(self, 
                graph1=None, 
                graph2=None,
                labels=None):

        gnn1_outputs, mask1 = self.gnn(graph1)
        gnn2_outputs, mask2 = self.gnn(graph2)
        gnn1_outputs = self.norm_graph(gnn1_outputs)
        gnn2_outputs = self.norm_graph(gnn2_outputs)

        if self.apply_mask == True:
            gnn1_outputs = gnn1_outputs * mask1
            gnn2_outputs = gnn2_outputs * mask2

        gnn_outputs = torch.cat((gnn1_outputs, gnn2_outputs), 1)
        
        pooled_output = self.dropout(gnn_outputs)
        logits = self.classifier(pooled_output)
        
        outputs = (logits,)

        if labels is not None:
            if self.data_type == "ddi_no_negative":
                weight = torch.tensor([17029.0/13008, 
                    17029.0/826, 
                    17029.0/1687, 
                    17029.0/1319, 
                    17029.0/189]).to('cuda' if torch.cuda.is_available() else 'cpu')
                loss_fct = CrossEntropyLoss(weight=weight)
            elif self.data_type == 'test':
                weight = torch.tensor([
                    3028.0/2049, 
                    3028.0/221, 
                    3028.0/360, 
                    3028.0/302, 
                    3028.0/96]).to('cuda' if torch.cuda.is_available() else 'cpu')
                loss_fct = CrossEntropyLoss(weight=weight)
            else:
                loss_fct = CrossEntropyLoss()
            labels = torch.nn.functional.one_hot(labels, 5).float()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.argmax(dim=-1))
            outputs = (loss,) + outputs

        return outputs

    def zero_init_params(self):
        self.update_cnt = 0
        for x in self.parameters():
            x.data *= 0

    def accumulate_params(self, model):
        self.update_cnt += 1
        for x, y in zip(self.parameters(), model.parameters()):
            x.data += y.data

    def average_params(self):
        for x in self.parameters():
            x.data /= self.update_cnt

    def restore_params(self):
        for x in self.parameters():
            x.data *= self.update_cnt
    
class Trainer:
    def __init__(self, 
                num_labels=5,
                dropout_rate=0.,
                activation="gelu",
                parameter_averaging=False,
                epsilon=1e-8,
                lr=1e-4,
                weight_decay=0,
                model_name_or_path="allenai/scibert_scivocab_uncased",
                wandb_available=False,
                freeze_bert=False,
                data_type="ddi_no_negative",
                log=True,
                optimizer='adamw',
                device='cuda',
                apply_mask=False,
                **kwargs):
        self.parameter_averaging = parameter_averaging
        self.lr = lr
        self.wandb_available = wandb_available
        self.freeze_bert = freeze_bert
        self.dropout_rate = dropout_rate
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
            dropout_rate=dropout_rate,
            activation=activation,
            weight_decay=weight_decay,
            data_type=data_type,
            config=self.config,
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
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, eps=epsilon)
        elif optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(optimizer_grouped_parameters, lr=lr, eps=epsilon)

    
    def train(self, training_loader, validation_loader, num_epochs,
                    train_mol1_loader=None, train_mol2_loader=None,
                    test_mol1_loader=None, test_mol2_loader=None,
                    filtered_lst_index=None, full_label=None,
                    modal='graph'):
        """ Train the model """
        train_loss = 0.0
        max_val_micro_f1 = 0.0
        nb_train_step = 0

        self.model.zero_grad()
        self.model.to(self.device)

        for epoch in range(num_epochs):
            print(f"================Epoch {epoch}================")
            epoch_iterator = zip(training_loader, train_mol1_loader, train_mol2_loader)
            for step, (batch, mol1, mol2) in enumerate(epoch_iterator):
                nb_train_step += 1
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'labels':         batch[5],
                          f'{modal}1': mol1.to(self.device),
                          f'{modal}2': mol2.to(self.device)}
                    
                outputs = self.model(**inputs)
                loss = outputs[0]
                loss.backward()
                
                train_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

                self.optimizer.step()
                self.model.zero_grad()

            train_loss = train_loss / nb_train_step
            self.train_loss.append(train_loss)
            
            results = self.evaluate(validation_loader,
                                    test_mol1_loader,
                                    test_mol2_loader,
                                    filtered_lst_index,
                                    full_label,
                                    option='val',
                                    modal=modal)
            
            # Save model checkpoint
            if results['Advise F'] > 0.1 and results['Effect F'] > 0.1 and results['Mechanism F'] > 0.1:
                if epoch > 5:
                    if not os.path.exists("checkpoints"):
                        os.makedirs("checkpoints")
                    save_model(f"checkpoints/{self.config.training_session_name}", f"epoch{epoch}val_micro_f1{results['microF']}.pt", \
                        self.config, self.model, self.wandb_available)
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
                 test_mol1_loader=None, 
                 test_mol2_loader=None,
                 filtered_lst_index=None,
                 full_label=None,
                 option='val',
                 modal=None):
        # Eval!
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for (batch, mol1, mol2) in zip(validation_loader, test_mol1_loader, test_mol2_loader):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'labels':         batch[5],
                          f'{modal}1': mol1.to(self.device),
                          f'{modal}2': mol2.to(self.device)}

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
            
        print(f"val_micro_f1: {self.val_micro_f1[-1]}\nval_macro_f1: {self.val_macro_f1[-1]}\nval_f_advise: {self.val_f_advise[-1]}\nval_f_effect: {self.val_f_effect[-1]}\nval_f_mechanism: {self.val_f_mechanism[-1]}\nval_f_int: {self.val_f_int[-1]}")    
            
        if self.log == True:
            wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                            y_true=out_label_ids, preds=preds,
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
        
        return result