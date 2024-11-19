import torch
import torch.nn

from bc5_2024.text.model.text_model import TextModel, BertModel
from bc5_2024.mol.gnn import GNN

class MultimodalModel(torch.nn.Module):
    def __init__(self, 
                 we,
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
                 target_class: int = 5,
                 num_node_features: int = 4, 
                 hidden_channels: int = 256,
                 text_model: str = 'bert',
                 modal: str = 'multimodal',
                 gnn_option: str = 'GATV2CONV',
                 num_layers_gnn: str = 3,
                 readout_option: str = 'global_max_pool',
                 activation_function: str = 'relu',
                 text_model_option: str = 'cnn',
                 position_embedding_type: str = 'linear',
                 device: str = 'cpu',
                 **kwargs):
        super(MultimodalModel, self).__init__()
        self.device = device

        if text_model == 'bert':
            self.text_model = BertModel(dropout_rate=dropout_rate,
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
                                        model_option=text_model_option,
                                        position_embedding_type=position_embedding_type,
                                        classifier=False,
                                        **kwargs).to(device)
        elif text_model == 'fasttext':
            self.text_model = TextModel(we=we,
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
                                        model_option=text_model_option,
                                        position_embedding_type=position_embedding_type,
                                        classifier=False,
                                        **kwargs).to(device)

        self.gnn1 = GNN(num_node_features=num_node_features,
                        hidden_channels=hidden_channels,
                        dropout_rate=dropout_rate, 
                        gnn_option=gnn_option,
                        num_layers_gnn=num_layers_gnn,
                        readout_option=readout_option,
                        activation_function=activation_function,
                        device=device).to(device)

        self.modal = modal
        
        if self.modal == '0':
            if text_model_option == 'bilstm':
                self.dense_to_tag = torch.nn.Linear(in_features=kwargs['lstm_hidden_size']*2, 
                                                    out_features=target_class,
                                                    bias=False)
            elif text_model_option == 'lstm':
                self.dense_to_tag = torch.nn.Linear(in_features=kwargs['lstm_hidden_size'], 
                                                    out_features=target_class,
                                                    bias=False)       
            else:
                self.dense_to_tag = torch.nn.Linear(in_features=conv1_out_channels+conv2_out_channels+conv3_out_channels, 
                                                    out_features=target_class,
                                                    bias=False)
        elif self.modal == '1':
            self.dense_to_tag = torch.nn.Linear(in_features=conv1_out_channels+conv2_out_channels+conv3_out_channels+hidden_channels, 
                                                out_features=target_class,
                                                bias=False)
        elif self.modal == '2':
            self.dense_to_tag = torch.nn.Linear(in_features=conv1_out_channels+conv2_out_channels+conv3_out_channels+600, 
                                                out_features=target_class,
                                                bias=False)
        elif self.modal == '3':
            self.dense_to_tag = torch.nn.Linear(in_features=conv1_out_channels+conv2_out_channels+conv3_out_channels+hidden_channels+600, 
                                                out_features=target_class,
                                                bias=False)
        elif self.modal == 'gnn_only':
            self.dense_to_tag = torch.nn.Linear(in_features=hidden_channels, 
                                                out_features=target_class,
                                                bias=False)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, 
                text_x, 
                mol_x1 = None, 
                mol_x1_smiles = None):
        if self.modal == '0':
            x = self.text_model(text_x)

            # Classifier
            x = self.dense_to_tag(x)
            x = self.softmax(x)

            return x
        elif self.modal == '1':
            text_x = self.text_model(text_x)
            mol_x1 = self.gnn1(mol_x1)

            x = torch.cat((text_x, mol_x1), dim=1)

            # Classifier
            x = self.dense_to_tag(x)
            x = self.softmax(x)

            return x
        elif self.modal == '2':
            text_x = self.text_model(text_x)
            
            mol_x1_smiles = mol_x1_smiles.squeeze(dim=1).float()

            x = torch.cat((text_x, mol_x1_smiles), dim=1)

            # Classifier
            x = self.dense_to_tag(x)
            x = self.softmax(x)

            return x
        elif self.modal == '3':
            text_x = self.text_model(text_x)
            mol_x1 = self.gnn1(mol_x1)
            
            mol_x1_smiles = mol_x1_smiles.squeeze(dim=1).float()
            
            x = torch.cat((text_x, mol_x1, mol_x1_smiles), dim=1)

            # Classifier
            x = self.dense_to_tag(x)
            x = self.softmax(x)

            return x
        elif self.modal == 'gnn_only':
            mol_x1 = self.gnn1(mol_x1)

            x = torch.cat((mol_x1), dim=1)

            # Classifier
            x = self.dense_to_tag(x)
            x = self.softmax(x)

            return x