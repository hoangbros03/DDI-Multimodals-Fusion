import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np
import wandb

from bc5_2024.multimodal.model import MultimodalModel
from bc5_2024.eval.bc5 import evaluate_bc5

class Trainer:
    def __init__(self,
                 we,
                 test_cand,
                 dropout_rate: float = 0.5,
                 word_embedding_size: int = 200,
                 tag_number: int = 51,
                 tag_embedding_size: int = 50,
                 position_number: int = 4,
                 position_embedding_size: int = 50,
                 direction_number: int = 3,
                 direction_embedding_size: int = 50,
                 edge_number: int = 46,
                 edge_embedding_size: int = 200,
                 token_embedding_size: int = 500,
                 dep_embedding_size: int = 500,
                 conv1_out_channels: int = 256,
                 conv2_out_channels: int = 256,
                 conv3_out_channels: int = 256,
                 conv1_length: int = 1,
                 conv2_length: int = 2,
                 conv3_length: int = 3,
                 target_class: int = 2,
                 num_node_features: int = 4, 
                 hidden_channels: int = 512,
                 lr: float = 0.0001,
                 weight_decay: float = 1e-4,
                 text_model: str = 'bert',
                 modal: str = 'multimodal',
                 gnn_option: str = 'GATVCONV2',
                 num_layers_gnn: str = 3,
                 readout_option: str = 'global_max_pool',
                 activation_function: str = 'relu',
                 text_model_option: str = 'cnn',
                 log: bool = True,
                 position_embedding_type: str = 'linear',
                 device: str = 'cpu',
                 **kwargs
                 ):
        self.model = MultimodalModel(we=we,
                                    dropout_rate=dropout_rate,
                                    word_embedding_size=word_embedding_size,
                                    tag_number=tag_number,
                                    tag_embedding_size=tag_embedding_size,
                                    position_number=position_number,
                                    position_embedding_size=position_embedding_size,
                                    direction_number=direction_number,
                                    direction_embedding_size=direction_embedding_size,
                                    edge_number=edge_number,
                                    edge_embedding_size=edge_embedding_size,
                                    token_embedding_size=token_embedding_size,
                                    dep_embedding_size=dep_embedding_size,
                                    conv1_out_channels=conv1_out_channels,
                                    conv2_out_channels=conv2_out_channels,
                                    conv3_out_channels=conv3_out_channels,
                                    conv1_length=conv1_length,
                                    conv2_length=conv2_length,
                                    conv3_length=conv3_length,
                                    target_class=target_class,
                                    num_node_features=num_node_features,
                                    hidden_channels=hidden_channels,
                                    text_model=text_model,
                                    modal=modal,
                                    gnn_option=gnn_option,
                                    num_layers_gnn=num_layers_gnn,
                                    readout_option=readout_option,
                                    activation_function=activation_function,
                                    text_model_option=text_model_option,
                                    device=device,
                                    position_embedding_type=position_embedding_type,
                                    **kwargs).to(device)
                                     
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = device
        self.test_cand = test_cand
        self.train_loss = list()
        self.val_loss = list()
        self.p = list()
        self.r = list()
        self.f = list()
        self.intra_p = list()
        self.intra_r = list()
        self.intra_f = list()
        self.inter_p = list()
        self.inter_r = list()
        self.inter_f = list()
        self.log = log
        
    def convert_label_to_2d(self, batch_label):
        i = 0
        for label in batch_label:
            i += 1
            if label == torch.tensor([0]).to(self.device):
                tmp = torch.tensor([1., 0.]).to(self.device)
            else:
                tmp = torch.tensor([0., 1.]).to(self.device)
            
            if i == 1:
                to_return = tmp.unsqueeze(0)
            else:
                to_return = torch.vstack((to_return, tmp))
        
        return to_return
    
    def train_one_epoch(self, 
                        train_loader_text, 
                        train_loader_mol1,
                        train_loader_mol1_bert):
        
        running_loss = 0.
        i = 0

        for ((a, batch_label), b, c) in zip(train_loader_text, 
                                            train_loader_mol1, 
                                            train_loader_mol1_bert):
            text = a.clone().detach().to(self.device)
            mol1 = b.to(self.device)
            mol1_bert = c.to(self.device)
            batch_label = batch_label.clone().detach().to(self.device)

            batch_label = self.convert_label_to_2d(batch_label)
            
            i += 1
            
            out = self.model(text, mol1, mol1_bert)
            self.optimizer.zero_grad()
            loss = self.criterion(out, batch_label)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            
        self.train_loss.append(running_loss)
        return running_loss
    
    def validate(self, 
                 val_loader_text, 
                 val_loader_mol1, 
                 val_loader_mol1_bert,
                 option):
        running_loss = 0.
        predictions = torch.tensor([]).to(self.device)
        labels = torch.tensor([]).to(self.device)
        
        with torch.no_grad():
            for ((a, batch_label), b, c) in zip(val_loader_text, 
                                                val_loader_mol1,
                                                val_loader_mol1_bert):
                text = a.clone().detach().to(self.device)
                mol1 = b.to(self.device)
                mol1_bert = c.to(self.device)

                batch_label = batch_label.clone().detach().to(self.device)

                out = self.model(text, mol1, mol1_bert)

                batch_label_for_loss = self.convert_label_to_2d(batch_label)
                loss = self.criterion(out, batch_label_for_loss)
                running_loss += loss.item()

                batch_prediction = torch.argmax(out, dim=1)

                predictions = torch.cat((predictions, batch_prediction))
                labels = torch.cat((labels, batch_label))
        
        labels = labels.squeeze()
        true_pred = []
        for i in range(len(labels)):
            if labels[i].cpu() == 1:
                true_pred.append(i)

        if self.log == True:
            wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                            y_true=labels.cpu().numpy(), preds=predictions.cpu().numpy(),
                            class_names=['0', '1'])})
        
        self.eval_bc5(predictions.cpu().numpy(), self.test_cand)

        if option == 'train':
            self.train_loss.append(running_loss)
        elif option == 'val':
            self.val_loss.append(running_loss)
        
    def train(self, train_loader_text, train_loader_mol1, train_loader_mol1_bert,
                    val_loader_text, val_loader_mol1, val_loader_mol1_bert, num_epochs):
        for epoch in tqdm(range(num_epochs), desc='Training...'):
            print(f"Epoch {epoch + 1} training...")
            running_loss = self.train_one_epoch(train_loader_text, 
                                                train_loader_mol1, 
                                                train_loader_mol1_bert)
            self.train_loss.append(running_loss)

            self.validate(val_loader_text, 
                          val_loader_mol1, 
                          val_loader_mol1_bert,
                          'val')
            
            if self.log == True:
                self.log_wandb()

    def eval_bc5(self, pred, cand):
        dct, lst = self.convert_pred_to_lst(pred, cand)
        return_tuple = evaluate_bc5(lst)
        self.p.append(return_tuple[0][0])
        self.r.append(return_tuple[0][1])
        self.f.append(return_tuple[0][2])
        self.intra_p.append(return_tuple[1][0])
        self.intra_r.append(return_tuple[1][1])
        self.intra_f.append(return_tuple[1][2])
        self.inter_p.append(return_tuple[2][0])
        self.inter_r.append(return_tuple[2][1])
        self.inter_f.append(return_tuple[2][2])
        return return_tuple

    def convert_pred_to_lst(self, pred, cand):
        dct = {}
        for i in range(len(pred)):
            if pred[i] == 1:
                if cand[i]['e1']['@id'] == '-1' or cand[i]['e2']['@id'] == '-1':
                    continue
                elif len(cand[i]['e1']['@id'].split('|')) > 1:
                    tmp = cand[i]['e1']['@id'].split('|')
                    for ent in tmp:
                        idx = cand[i]['id'].split('_')[0]
                        if idx in dct.keys():
                            if f"{ent}_{cand[i]['e2']['@id']}" not in dct[idx]:
                                dct[idx].append(f"{ent}_{cand[i]['e2']['@id']}")
                        else:
                            dct[idx] = [f"{ent}_{cand[i]['e2']['@id']}"]
                elif len(cand[i]['e2']['@id'].split('|')) > 1:
                    tmp = cand[i]['e2']['@id'].split('|')
                    for ent in tmp:
                        idx = cand[i]['id'].split('_')[0]
                        if idx in dct.keys():
                            if f"{cand[i]['e1']['@id']}_{ent}" not in dct[idx]:
                                dct[idx].append(f"{cand[i]['e1']['@id']}_{ent}")
                        else:
                            dct[idx] = [f"{cand[i]['e1']['@id']}_{ent}"]
                else:
                    idx = cand[i]['id'].split('_')[0]
                    if idx in dct.keys():
                        if f"{cand[i]['e1']['@id']}_{cand[i]['e2']['@id']}" not in dct[idx]:
                            dct[idx].append(f"{cand[i]['e1']['@id']}_{cand[i]['e2']['@id']}")
                    else:
                        dct[idx] = [f"{cand[i]['e1']['@id']}_{cand[i]['e2']['@id']}"]

        lst = []
        for k, v in dct.items():
            for _ in v:
                lst.append((k, _, "CID"))

        return dct, lst
            
    def log_wandb(self):
        wandb.log(
            {
                "train_loss": self.train_loss[-1],
                "val_loss": self.val_loss[-1],
                "precision": self.p[-1],
                "recall": self.r[-1],
                "f1": self.f[-1],
                "intra_precision": self.intra_p[-1],
                "intra_recall": self.intra_r[-1],
                "intra_f1": self.intra_f[-1],
                "inter_precision": self.inter_p[-1],
                "inter_recall": self.inter_r[-1],
                "inter_f1": self.inter_f[-1]
            }
        )