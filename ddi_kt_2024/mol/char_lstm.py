import torch
import torch.nn as nn

class CharLSTM(nn.Module):
    def __init__(self,
                 embedding: bool = True,
                 num_chars: int = 41,
                 input_dim: int = 16,
                 hidden_dim: int = 16,
                 output_dim: int = 16,
                 device: str = 'cpu'):
        super(CharLSTM, self).__init__()
        self.device = device
        self.num_chars = num_chars
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding = embedding
        
        if self.embedding:
            self.character_encoder = nn.Embedding(num_embeddings=num_chars+1, embedding_dim=input_dim, padding_idx=0)

        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=1, 
                            batch_first=True, 
                            bidirectional=True)
        
        self.linear = nn.Linear(2*hidden_dim, output_dim, bias=False)

    def forward(self, x):
        if self.embedding:
            x = self.character_encoder(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)

        return x