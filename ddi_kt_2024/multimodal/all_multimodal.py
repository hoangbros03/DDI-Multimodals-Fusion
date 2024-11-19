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
    def __init__(self,
                num_labels,
                dropout_rate_other,
                dropout_rate_bert_output,
                out_conv_list_dim,
                conv_window_size: list,
                max_seq_length,
                middle_layer_size,
                model_name_or_path,
                pos_emb_dim=10,
                activation="gelu",
                weight_decay=0.0,
                freeze_bert=True,
                data_type="ddi_no_negative",
                config=None,
                image_reduced_dim=20,
                loss_type="normal",
                apply_mask=False,
                **kwargs
                ):
        if 'seed' in kwargs:
            seed = kwargs['seed']
            print(f"Setting seed to {seed}")
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"Dropout text will be {dropout_rate_bert_output}, while dropout of other modal is {dropout_rate_other}")
        print(f"Loss type is {loss_type}")
        super(BertForSequenceClassification, self).__init__(config)
        self.data_type=data_type
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
        self.data_type = data_type
        self.apply_mask = apply_mask

        # CNN
        self.conv_list = nn.ModuleList([nn.Conv1d(768+2*self.pos_emb_dim, out_conv_list_dim, w, padding=(w-1)//2) for w in self.conv_window_size])
        self.pos_emb = nn.Embedding(2*self.max_seq_length, self.pos_emb_dim, padding_idx=0)
        self.norm_text = torch.nn.BatchNorm1d(num_features=len(self.conv_window_size)*self.out_conv_list_dim)

        # Graph
        self.gnn = GNN(**kwargs)
        self.norm_graph = torch.nn.BatchNorm1d(self.gnn.hidden_channels)
        print(f"Using Graph. Output dims = {self.gnn.hidden_channels}")

        self.freeze_gnn = kwargs['freeze_gnn']
        if self.freeze_gnn:
            print("Freezing GNN")
            for param in self.gnn.parameters():
                param.requires_grad = False

        # Formula
        self.formula_conv_output_size = kwargs['formula_conv_output_size']
        self.formula_conv_window_size = kwargs['formula_conv_window_size']
        self.formula_clf = BertForFormulaClassification(**kwargs)
        self.norm_formula = torch.nn.BatchNorm1d(num_features=self.formula_conv_output_size)
        self.formula_conv = nn.Conv1d(768, self.formula_conv_output_size, self.formula_conv_window_size, padding=(self.formula_conv_window_size-1)//2)
        print(f"Using Formula. Output dims = {self.formula_conv_output_size}")

        self.freeze_formula = kwargs['freeze_formula']
        if self.freeze_formula:
            print("Freezing Formula")
            for param in self.formula_clf.parameters():
                param.requires_grad = False
            for param in self.formula_conv.parameters():
                param.requires_grad = False

        # Desc
        self.desc_conv_output_size = kwargs['desc_conv_output_size']
        self.desc_conv_window_size = kwargs['desc_conv_window_size']
        self.desc_layer_hidden = kwargs['desc_layer_hidden']
        print(f"Using descriptions. Output dims = {self.desc_conv_output_size}")
        self.desc_conv = nn.Conv1d(768, self.desc_conv_output_size, self.desc_conv_window_size, padding=(self.desc_conv_window_size-1)//2)
        self.norm_desc = torch.nn.BatchNorm1d(num_features=self.desc_conv_output_size)

        # Image
        print(f"Using image classifier from 2000 (fixed) to {image_reduced_dim} dimensions. As 2000 is fixed so change as need")
        self.image_classifier_dim = 2000
        self.reduced_dim = image_reduced_dim
        self.reduce_img_dim_layer = nn.Linear(self.image_classifier_dim, self.reduced_dim, bias=True)
        self.batch_norm_img = nn.BatchNorm1d(num_features = self.reduced_dim)
    
        self.init_weights()
        self.pos_emb.weight.data.uniform_(-1e-3, 1e-3)
    
        self.bert = transformers.BertModel.from_pretrained(self.model_name_or_path)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        if self.middle_layer_size != 0:
            self.middle_classifier = nn.Linear(self.reduced_dim + self.desc_conv_output_size*2 + \
                                               self.formula_conv_output_size*2 + self.gnn.hidden_channels*2 + len(self.conv_window_size)*self.out_conv_list_dim,
                                              self.middle_layer_size)
            self.classifier = nn.Linear(self.middle_layer_size, self.num_labels)
        else:
            self.classifier = nn.Linear(self.reduced_dim + self.desc_conv_output_size*2 + \
                                               self.formula_conv_output_size*2 + self.gnn.hidden_channels*2 + len(self.conv_window_size)*self.out_conv_list_dim, self.num_labels)
        
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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None,
                relative_dist1=None, relative_dist2=None,
                formula1=None, formula2=None,
                graph1=None, graph2=None,
                desc1=None, desc2=None,
                image_classifier_output1=None,
                labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        # CNN
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

        # Graph
        gnn1_outputs, mask1 = self.gnn(graph1)
        gnn2_outputs, mask2 = self.gnn(graph2)
        gnn1_outputs = self.norm_graph(gnn1_outputs)
        gnn2_outputs = self.norm_graph(gnn2_outputs)

        if self.apply_mask == True:
            gnn1_outputs = gnn1_outputs * mask1
            gnn2_outputs = gnn2_outputs * mask2

        gnn_outputs = self.modal_dropout(torch.cat((gnn1_outputs, gnn2_outputs), 1))

        # Formula
        formula1_outputs, mask1 = self.formula_clf(**formula1)
        formula2_outputs, mask2 = self.formula_clf(**formula2)
        formula1_conv_output = self.activation(self.formula_conv(formula1_outputs.transpose(1,2)))
        formula2_conv_output = self.activation(self.formula_conv(formula2_outputs.transpose(1,2)))
        pooled_formula1_output, _ = torch.max(formula1_conv_output, -1)
        pooled_formula2_output, _ = torch.max(formula2_conv_output, -1)
        pooled_formula1_output = self.norm_formula(pooled_formula1_output)
        pooled_formula2_output = self.norm_formula(pooled_formula2_output)

        if self.apply_mask == True:
            pooled_formula1_output = pooled_formula1_output * mask1
            pooled_formula2_output = pooled_formula2_output * mask2

        formula_output = self.modal_dropout(torch.cat((pooled_formula1_output, pooled_formula2_output), 1))
        
        # Desc
        desc1_conv_input = desc1
        desc2_conv_input = desc2
        desc1_conv_output = self.activation(self.desc_conv(desc1_conv_input.transpose(1,2)))
        desc2_conv_output = self.activation(self.desc_conv(desc2_conv_input.transpose(1,2)))
        pooled_desc1_output, _ = torch.max(desc1_conv_output, -1)
        pooled_desc2_output, _ = torch.max(desc2_conv_output, -1)
        pooled_desc1_output = self.norm_desc(pooled_desc1_output)
        pooled_desc2_output = self.norm_desc(pooled_desc2_output)
        desc_outputs = self.modal_dropout(torch.cat((pooled_desc1_output, pooled_desc2_output), 1))

        # Image
        image_classifier_output1 = self.activation(self.reduce_img_dim_layer(image_classifier_output1))
        image_classifier_output1 = self.modal_dropout(self.batch_norm_img(image_classifier_output1))

        # Fusion
        pooled_output = torch.cat((pooled_output, desc_outputs, formula_output, gnn_outputs, image_classifier_output1), 1)

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
    

class Trainer:
    def __init__(self, 
            num_labels=5,
            dropout_rate_other=0.3,
            dropout_rate_bert_output=0.5,
            out_conv_list_dim=128,
            conv_window_size=[3, 5, 7],
            max_seq_length=256,
            pos_emb_dim=10,
            middle_layer_size=256,
            activation="gelu",
            parameter_averaging=False,
            epsilon=1e-8,
            lr=1e-4,
            optimizer='rmsprop',
            weight_decay=0,
            model_name_or_path="allenai/scibert_scivocab_uncased",
            wandb_available=False,
            freeze_bert=False,
            data_type="ddi_no_negative",
            log=True,
            device='cuda',
            image_reduced_dim=20,
            apply_mask=False,
            **kwargs):
        self.parameter_averaging = parameter_averaging
        self.lr = lr
        self.wandb_available = wandb_available
        self.freeze_bert = freeze_bert
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
            data_type=data_type,
            config=self.config,
            image_reduced_dim=image_reduced_dim,
            apply_mask=apply_mask,
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

    def train(self, training_loader, validation_loader, num_epochs,
                    train_mol1_loader=None, train_mol2_loader=None,
                    test_mol1_loader=None, test_mol2_loader=None,
                    train_mol3_loader=None, train_mol4_loader=None,
                    test_mol3_loader=None, test_mol4_loader=None,
                    train_mol5_loader=None, train_mol6_loader=None,
                    test_mol5_loader=None, test_mol6_loader=None,
                    train_image_loader=None, test_image_loader=None,
                    filtered_lst_index=None, full_label=None,
                    modal1='graph', modal2='formula', modal3='desc', **kwargs):
        """ Train the model """
        print("modal1='graph', modal2='formula', modal3='desc' and image.\n Graph, formula, and description separate into 2 entities as normally.\n Image is a set of 2000-dim features.")
        train_loss = 0.0
        max_val_micro_f1 = 0.0
        nb_train_step = 0

        self.model.zero_grad()
        self.model.to(self.device)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=num_epochs
        )

        for epoch in range(num_epochs):
            print(f"================Epoch {epoch}================")
            epoch_iterator = zip(
                training_loader, 
                train_mol1_loader, 
                train_mol2_loader, 
                train_mol3_loader, 
                train_mol4_loader,
                train_mol5_loader,
                train_mol6_loader,
                train_image_loader)
            for step, (batch, mol1, mol2, mol3, mol4, mol5, mol6, image) in enumerate(tqdm(epoch_iterator)):
                nb_train_step += 1
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'relative_dist1': batch[3],
                          'relative_dist2': batch[4],
                          'labels':         batch[5],
                          f'{modal1}1': mol1.to(self.device),
                          f'{modal1}2': mol2.to(self.device),
                          f'{modal2}1': mol3.to(self.device),
                          f'{modal2}2': mol4.to(self.device),
                          f'{modal3}1': mol5.to(self.device),
                          f'{modal3}2': mol6.to(self.device),
                          'image_classifier_output1': image[5].to(self.device)}
                    
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
            
            results = self.evaluate(validation_loader,
                                    test_mol1_loader,
                                    test_mol2_loader,
                                    test_mol3_loader,
                                    test_mol4_loader,
                                    test_mol5_loader,
                                    test_mol6_loader,
                                    test_image_loader,
                                    filtered_lst_index,
                                    full_label,
                                    option='val',
                                    modal1=modal1,
                                    modal2=modal2,
                                    modal3=modal3)
            
            # Save model checkpoint
            if max_val_micro_f1 < results['microF']:
                if not os.path.exists("checkpoints"):
                    os.makedirs("checkpoints")
                save_model(f"checkpoints", f"epoch{epoch}val_micro_f1{results['microF']}.pt", \
                    None, self.model, True)
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
                 test_mol3_loader=None,
                 test_mol4_loader=None,
                 test_mol5_loader=None,
                 test_mol6_loader=None,
                 test_image_loader=None,
                 filtered_lst_index=None,
                 full_label=None,
                 option='val',
                 modal1=None,
                 modal2=None,
                 modal3=None):
        # Eval!
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for (batch, mol1, mol2, mol3, mol4, mol5, mol6, image) in tqdm(zip(validation_loader, test_mol1_loader, test_mol2_loader, test_mol3_loader, test_mol4_loader, test_mol5_loader, test_mol6_loader, test_image_loader)):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'relative_dist1': batch[3],
                          'relative_dist2': batch[4],
                          'labels':         batch[5],
                          f'{modal1}1': mol1.to(self.device),
                          f'{modal1}2': mol2.to(self.device),
                          f'{modal2}1': mol3.to(self.device),
                          f'{modal2}2': mol4.to(self.device),
                          f'{modal3}1': mol5.to(self.device),
                          f'{modal3}2': mol6.to(self.device),
                          'image_classifier_output1': image[5].to(self.device)}

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
        
        return result