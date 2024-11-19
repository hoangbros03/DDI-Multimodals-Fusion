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

from ddi_kt_2024.utils import save_model
from ddi_kt_2024.multimodal.bert import *

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self,
                num_labels,
                dropout_rate_other,
                middle_layer_size,
                activation="gelu",
                data_type="ddi_no_negative",
                config=None,
                **kwargs
                ):
        super(BertForSequenceClassification, self).__init__(config)
        self.data_type=data_type
        self.num_labels = num_labels
        self.dropout_rate_other = dropout_rate_other
        self.modal_dropout = nn.Dropout(dropout_rate_other)

        activations = {'relu':nn.ReLU(), 'elu':nn.ELU(), 'leakyrelu':nn.LeakyReLU(), 'prelu':nn.PReLU(),
                       'relu6':nn.ReLU6, 'rrelu':nn.RReLU(), 'selu':nn.SELU(), 'celu':nn.CELU(), 'gelu':nn.GELU()}
        self.activation = activations[activation]
        self.middle_layer_size = middle_layer_size

        # Desc
        self.desc_conv_output_size = kwargs['desc_conv_output_size']
        self.desc_conv_window_size = kwargs['desc_conv_window_size']
        self.desc_layer_hidden = kwargs['desc_layer_hidden']
        print(f"Using descriptions. Output dims = {self.desc_conv_output_size}")
        self.desc_conv = nn.Conv1d(768, self.desc_conv_output_size, self.desc_conv_window_size, padding=(self.desc_conv_window_size-1)//2)
    
        self.init_weights()

        if self.middle_layer_size != 0:
            self.middle_classifier = nn.Linear(self.desc_conv_output_size*2, self.middle_layer_size)
            self.classifier = nn.Linear(self.middle_layer_size, self.num_labels)
        else:
            self.classifier = nn.Linear(self.desc_conv_output_size*2, self.num_labels)
        
        self._init_weights(self.classifier)
            

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

    def forward(self, desc1=None, desc2=None, labels=None):
        # Desc
        desc1_conv_input = desc1
        desc2_conv_input = desc2
        desc1_conv_output = self.activation(self.desc_conv(desc1_conv_input.transpose(1,2)))
        desc2_conv_output = self.activation(self.desc_conv(desc2_conv_input.transpose(1,2)))
        pooled_desc1_output, _ = torch.max(desc1_conv_output, -1)
        pooled_desc2_output, _ = torch.max(desc2_conv_output, -1)
        desc_outputs = self.modal_dropout(torch.cat((pooled_desc1_output, pooled_desc2_output), 1))

        if self.middle_layer_size == 0:
            logits = self.classifier(desc_outputs)
        else:
            middle_output = self.activation(self.middle_classifier(desc_outputs))
            logits = self.classifier(middle_output)
        
#         outputs = (logits,) + outputs[2:]
        
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
        outputs = tuple((loss, logits))

        return outputs
    

class Trainer:
    def __init__(self, 
            num_labels=5,
            dropout_rate_other=0.3,
            middle_layer_size=256,
            activation="gelu",
            parameter_averaging=False,
            epsilon=1e-8,
            lr=1e-4,
            optimizer='rmsprop',
            weight_decay=0,
            model_name_or_path="allenai/scibert_scivocab_uncased",
            wandb_available=False,
            data_type="ddi_no_negative",
            log=True,
            device='cuda',
            scheduler=False,
            **kwargs):
        self.parameter_averaging = parameter_averaging
        self.lr = lr
        self.wandb_available = wandb_available
        self.config = BertConfig.from_pretrained(model_name_or_path)
        self.train_loss = list()
        self.train_loss_all = list()
        self.train_micro_f1 = list()
        self.val_loss = list()
        self.val_loss_text = list()
        self.val_loss_text_graph = list()
        self.val_loss_text_formula = list()
        self.val_loss_all = list()
        self.val_micro_f1 = list()
        self.val_macro_f1 = list()
        self.val_precision = list()
        self.val_recall = list()
        self.val_f_mechanism = list()
        self.val_f_effect = list()
        self.val_f_advise = list()
        self.val_f_int = list()
        self.val_r_mechanism = list()
        self.val_r_effect = list()
        self.val_r_advise = list()
        self.val_r_int = list()
        self.val_p_mechanism = list()
        self.val_p_effect = list()
        self.val_p_advise = list()
        self.val_p_int = list()
        self.log = log
        self.device = device

        self.model = BertForSequenceClassification(
            num_labels=num_labels,
            dropout_rate_other=dropout_rate_other,
            middle_layer_size=middle_layer_size,
            model_name_or_path=model_name_or_path,
            activation=activation,
            data_type=data_type,
            config=self.config,
            **kwargs
        )
        
        self.weight_decay = weight_decay
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # self.optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
        if optimizer == 'adamw':
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=epsilon)
        elif optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(optimizer_grouped_parameters, lr=lr, eps=epsilon)
        
        self.use_scheduler = scheduler

    def train(self, num_epochs, training_loader, validation_loader,
                    train_mol1_loader=None, train_mol2_loader=None,
                    test_mol1_loader=None, test_mol2_loader=None,
                    filtered_lst_index=None, full_label=None, modal='graph', **kwargs):
        """ Train the model """
        print("Graph only")
        train_loss = 0.0
        max_val_micro_f1 = 0.0
        nb_train_step = 0

        self.model.zero_grad()
        self.model.to(self.device)

        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=num_epochs
            )

        for epoch in range(num_epochs):
            print(f"================Epoch {epoch}================")
            epoch_iterator = zip(
                training_loader,
                train_mol1_loader, 
                train_mol2_loader)
            for step, (batch, mol1, mol2) in enumerate(tqdm(epoch_iterator)):
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

            if self.use_scheduler:
                self.scheduler.step()
                print(f"LR: {self.scheduler.get_last_lr()}")

            train_loss = train_loss / nb_train_step
            self.train_loss.append(train_loss)
            
            results, pred_logit = self.evaluate(validation_loader,
                                    test_mol1_loader,
                                    test_mol2_loader,
                                    filtered_lst_index,
                                    full_label,
                                    option='val',
                                    modal=modal)
            
            # Save model checkpoint
            if max_val_micro_f1 < results['microF']:
                if not os.path.exists("checkpoints"):
                    os.makedirs("checkpoints")
                save_model(
                        output_path=f"checkpoints", 
                        file_name=f"epoch{epoch}val_micro_f1{results['microF']}.pt",
                        config=self.config, 
                        model=self.model, 
                        pred_logit=pred_logit, 
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
        for (batch, mol1, mol2) in tqdm(zip(validation_loader, test_mol1_loader, test_mol2_loader)):
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
        pred_logit = preds
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
            self.val_p_mechanism.append(result['Mechanism Precision'])
            self.val_r_mechanism.append(result['Mechanism Recall'])
            self.val_f_effect.append(result['Effect F'])
            self.val_p_effect.append(result['Effect Precision'])
            self.val_r_effect.append(result['Effect Recall'])
            self.val_f_advise.append(result['Advise F'])
            self.val_p_advise.append(result['Advise Precision'])
            self.val_r_advise.append(result['Advise Recall'])
            self.val_f_int.append(result['Int. F'])
            self.val_p_int.append(result['Int. Precision'])
            self.val_r_int.append(result['Int. Recall'])
            
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
                        "val_f_int": self.val_f_int[-1],
                        "val_p_advise": self.val_p_advise[-1],
                        "val_p_effect": self.val_p_effect[-1],
                        "val_p_mechanism": self.val_p_mechanism[-1],
                        "val_p_int": self.val_p_int[-1],
                        "val_r_advise": self.val_r_advise[-1],
                        "val_r_effect": self.val_r_effect[-1],
                        "val_r_mechanism": self.val_r_mechanism[-1],
                        "val_r_int": self.val_r_int[-1],})
        
        return result, pred_logit