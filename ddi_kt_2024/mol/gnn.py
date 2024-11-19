import torch
import torch.nn as nn
from torch_geometric.nn.models import GAT, GCN, NeuralFingerprint
from torch_geometric.nn import (
    aggr,
    global_mean_pool, 
    global_max_pool, 
    global_add_pool
)

class GNN(torch.nn.Module):
    def __init__(self, 
                 atom_embedding_dim: int = 12,
                 bond_embedding_dim: int = 8,
                 embed_to_dim: int = 3,
                 hidden_channels: int = 32,
                 dropout_rate: float = 0.1,
                 num_layers_gnn: int = 5,
                 gnn_option: str = 'GAT',
                 readout_option: str = 'transformer',
                 device: str = 'cpu',
                 use_edge_attr: bool = True,
                 use_gat_v2: bool = False,
                 **kwargs):
        
        super(GNN, self).__init__()
        self.device = device
        self.dropout = dropout_rate
        self.hidden_channels = hidden_channels
        self.embed_to_dim = embed_to_dim
        self.num_layers_gnn = num_layers_gnn
        self.embed_to_dim = embed_to_dim
        self.gnn_option = gnn_option
        self.use_edge_attr = use_edge_attr
        self.use_gat_v2 = use_gat_v2
        self.bool_dim = 2

        self.atomic_num_embedding = nn.Embedding(num_embeddings=83+1, embedding_dim=atom_embedding_dim, padding_idx=0)
        self.degree_embedding = nn.Embedding(num_embeddings=4+1, embedding_dim=self.embed_to_dim, padding_idx=0)
        self.num_implicit_hs_embedding = nn.Embedding(num_embeddings=4+1, embedding_dim=self.embed_to_dim, padding_idx=0)
        self.explicit_valence_embedding = nn.Embedding(num_embeddings=7+1, embedding_dim=self.embed_to_dim, padding_idx=0)
        self.implicit_valence_embedding = nn.Embedding(num_embeddings=4+1, embedding_dim=self.embed_to_dim, padding_idx=0)
        self.total_valence_embedding = nn.Embedding(num_embeddings=7+1, embedding_dim=self.embed_to_dim, padding_idx=0)
        self.num_radical_electrons = nn.Embedding(num_embeddings=4+1, embedding_dim=self.embed_to_dim, padding_idx=0)
        self.hybridization_embedding = nn.Embedding(num_embeddings=6+1, embedding_dim=self.embed_to_dim, padding_idx=0)
        self.is_aromatic_embedding = nn.Embedding(num_embeddings=2+1, embedding_dim=self.bool_dim, padding_idx=2)
        self.is_in_ring_embedding = nn.Embedding(num_embeddings=2+1, embedding_dim=self.bool_dim, padding_idx=2)

        self.bond_type_embedding = nn.Embedding(num_embeddings=12+1, embedding_dim=bond_embedding_dim, padding_idx=0)
        self.bond_stereo_embedding = nn.Embedding(num_embeddings=3+1, embedding_dim=self.embed_to_dim, padding_idx=0)
        self.is_conjugated_embedding = nn.Embedding(num_embeddings=2+1, embedding_dim=self.bool_dim, padding_idx=2)
        self.is_aromatic_bond_embedding = nn.Embedding(num_embeddings=2+1, embedding_dim=self.bool_dim, padding_idx=2)
        self.is_in_ring_bond_embedding = nn.Embedding(num_embeddings=2+1, embedding_dim=self.bool_dim, padding_idx=2)

        self.atomic_num_embedding.weight.data.uniform_(-1e-3, 1e-3)
        self.degree_embedding.weight.data.uniform_(-1e-3, 1e-3)
        self.num_implicit_hs_embedding.weight.data.uniform_(-1e-3, 1e-3)
        self.explicit_valence_embedding.weight.data.uniform_(-1e-3, 1e-3)
        self.implicit_valence_embedding.weight.data.uniform_(-1e-3, 1e-3)
        self.total_valence_embedding.weight.data.uniform_(-1e-3, 1e-3)
        self.num_radical_electrons.weight.data.uniform_(-1e-3, 1e-3)
        self.hybridization_embedding.weight.data.uniform_(-1e-3, 1e-3)
        self.is_aromatic_embedding.weight.data.uniform_(-1e-3, 1e-3)
        self.is_in_ring_embedding.weight.data.uniform_(-1e-3, 1e-3)
        
        self.bond_type_embedding.weight.data.uniform_(-1e-3, 1e-3)
        self.bond_stereo_embedding.weight.data.uniform_(-1e-3, 1e-3)
        self.is_conjugated_embedding.weight.data.uniform_(-1e-3, 1e-3)
        self.is_aromatic_bond_embedding.weight.data.uniform_(-1e-3, 1e-3)
        self.is_in_ring_bond_embedding.weight.data.uniform_(-1e-3, 1e-3)
        
        self.norm_node = torch.nn.BatchNorm1d(atom_embedding_dim+7*self.embed_to_dim+2*self.bool_dim)
        self.norm_edge = torch.nn.BatchNorm1d(bond_embedding_dim+1*self.embed_to_dim+3*self.bool_dim)

        if gnn_option == 'GAT':
            self.gnn = GAT(in_channels=atom_embedding_dim+7*self.embed_to_dim+2*self.bool_dim, 
                           num_layers=self.num_layers_gnn,
                           hidden_channels=self.hidden_channels,
                           out_channels=self.hidden_channels,
                           dropout=self.dropout,
                           v2=self.use_gat_v2)
        elif gnn_option == 'GCN':
            self.gnn = GCN(in_channels=atom_embedding_dim+7*self.embed_to_dim+2*self.bool_dim, 
                           num_layers=self.num_layers_gnn,
                           hidden_channels=self.hidden_channels,
                           out_channels=self.hidden_channels,
                           dropout=self.dropout)
        elif gnn_option == 'NEURALFINGERPRINT':
            self.gnn = NeuralFingerprint(
                in_channels=atom_embedding_dim+7*self.embed_to_dim+2*self.bool_dim,
                hidden_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                num_layers=self.num_layers_gnn
            )
            
        if readout_option == 'global_max_pool':
            self.readout_layer = global_max_pool
        elif readout_option == 'global_mean_pool':
            self.readout_layer = global_mean_pool
        elif readout_option == 'global_add_pool':
            self.readout_layer = global_add_pool
        elif readout_option == 'softmax':
            self.readout_layer = aggr.SoftmaxAggregation(learn=True)
        elif readout_option == 'sort':
            self.readout_layer = aggr.SortAggregation(k=2)
        elif readout_option == 'lstm':
            self.readout_layer = aggr.LSTMAggregation(in_channels=self.gnn.out_channels,
                                                      out_channels=self.gnn.out_channels)
        elif readout_option == 'transformer':
            self.readout_layer = aggr.SetTransformerAggregation(channels=self.gnn.out_channels,
                                                                num_encoder_blocks=4,
                                                                num_decoder_blocks=4,
                                                                heads=4)

    def forward(self, mol):
        x, edge_index, edge_attr, batch, smiles = mol.x, mol.edge_index, mol.edge_attr, mol.batch, mol.smiles
        
        mask = list()
        for i in smiles:
            if i == 'None':
                mask.append(0)
            else:
                mask.append(1)
                
        mask = torch.FloatTensor(mask).unsqueeze(dim=1).to(x.device)
        
        atom_feature0 = self.atomic_num_embedding(x[:, 0].int())
        atom_feature1 = self.degree_embedding(x[:, 1].int())
        atom_feature2 = self.num_implicit_hs_embedding(x[:, 2].int())
        atom_feature3 = self.explicit_valence_embedding(x[:, 3].int())
        atom_feature4 = self.implicit_valence_embedding(x[:, 4].int())
        atom_feature5 = self.total_valence_embedding(x[:, 5].int())
        atom_feature6 = self.num_radical_electrons(x[:, 6].int())
        atom_feature7 = self.hybridization_embedding(x[:, 7].int())
        atom_feature8 = self.is_aromatic_embedding(x[:, 8].int())
        atom_feature9 = self.is_in_ring_embedding(x[:, 9].int())

        bond_feature0 = self.bond_type_embedding(edge_attr[:, 0].int())
        bond_feature1 = self.bond_stereo_embedding(edge_attr[:, 1].int())
        bond_feature2 = self.is_conjugated_embedding(edge_attr[:, 2].int())
        bond_feature3 = self.is_aromatic_bond_embedding(edge_attr[:, 3].int())
        bond_feature4 = self.is_in_ring_bond_embedding(edge_attr[:, 4].int())


        x = torch.cat([atom_feature0,
                       atom_feature1,
                       atom_feature2,
                       atom_feature3,
                       atom_feature4,
                       atom_feature5,
                       atom_feature6,
                       atom_feature7,
                       atom_feature8,
                       atom_feature9], dim=1)
        
        edge_attr = torch.cat([bond_feature0,
                               bond_feature1,
                               bond_feature2,
                               bond_feature3,
                               bond_feature4], dim=1)
        
        x = self.norm_node(x)
        edge_attr = self.norm_edge(edge_attr)
        
        # GNN pass
        if self.gnn_option == 'NEURALFINGERPRINT':
            x = self.gnn(x=x,
                         edge_index=edge_index,
                         batch=batch)
        elif not self.use_edge_attr:
            x = self.gnn(x=x,
                         edge_index=edge_index,
                         batch=batch)
            x = self.readout_layer(x, batch)
        elif self.use_edge_attr:
            x = self.gnn(x=x, 
                         edge_index=edge_index,
                         edge_attr=edge_attr,
                         batch=batch)
            x = self.readout_layer(x, batch)
        
        return x, mask


