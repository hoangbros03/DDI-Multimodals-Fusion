import torch
import torch.nn as nn

class TextModel(nn.Module):
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
                 model_option: str = 'cnn',
                 position_embedding_type: str = 'linear',
                 classifier: bool = False,
                 device: str = 'cpu',
                 **kwargs
                 ):

        super(TextModel, self).__init__()
        self.classifier = classifier
        self.model_option = model_option
        self.position_embedding_size = position_embedding_size
        self.position_embedding_type = position_embedding_type
        self.device = device

        if self.model_option == 'lstm' or self.model_option == 'bilstm':
            self.lstm_hidden_size = kwargs['lstm_hidden_size']
            self.lstm_num_layers = kwargs['lstm_num_layers']


        self.w2v = nn.Embedding.from_pretrained(torch.tensor(we.vectors))
        self.tag_embedding = nn.Embedding(tag_number, tag_embedding_size, padding_idx=0)
        self.direction_embedding = nn.Embedding(direction_number, direction_embedding_size, padding_idx=0)
        self.edge_embedding = nn.Embedding(edge_number, edge_embedding_size, padding_idx=0)

        self.normalize_position = nn.Linear(in_features=position_number,
                                            out_features=position_embedding_size,
                                            bias=False)
        
        self.normalize_tokens = nn.Linear(in_features=word_embedding_size+tag_embedding_size+position_embedding_size,
                                          out_features=token_embedding_size,
                                          bias=False)
        
        self.normalize_dep = nn.Linear(in_features=direction_embedding_size+edge_embedding_size,
                                       out_features=dep_embedding_size,
                                       bias=False)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        if self.model_option == 'cnn':
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=1,
                        out_channels=conv1_out_channels,
                        kernel_size=(conv1_length, token_embedding_size * 2 + dep_embedding_size),
                        stride=1,
                        bias=False),
                nn.ReLU()
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=1,
                        out_channels=conv2_out_channels,
                        kernel_size=(conv2_length, token_embedding_size * 2 + dep_embedding_size),
                        stride=1,
                        bias=False),
                nn.ReLU()
            )

            self.conv3 = nn.Sequential(
                nn.Conv2d(in_channels=1,
                        out_channels=conv3_out_channels,
                        kernel_size=(conv3_length, token_embedding_size * 2 + dep_embedding_size),
                        stride=1,
                        bias=False),
                nn.ReLU()
            )
            self.dense_to_tag = nn.Linear(in_features=conv1_out_channels + conv2_out_channels + conv3_out_channels,
                                      out_features=target_class,
                                      bias=False)
        elif self.model_option == 'lstm':
            self.lstm = nn.LSTM(input_size=token_embedding_size * 2 + dep_embedding_size,
                                hidden_size=self.lstm_hidden_size,
                                num_layers=self.lstm_num_layers,
                                batch_first=True,
                                bidirectional=False,
                                dropout=dropout_rate)
            self.dense_to_tag = nn.Linear(in_features=self.lstm_hidden_size,
                                        out_features=target_class,
                                        bias=False)
        elif self.model_option == 'bilstm':
            self.lstm = nn.LSTM(input_size=token_embedding_size * 2 + dep_embedding_size,
                                hidden_size=self.lstm_hidden_size,
                                num_layers=self.lstm_num_layers,
                                batch_first=True,
                                bidirectional=True,
                                dropout=dropout_rate)
            self.dense_to_tag = nn.Linear(in_features=self.lstm_hidden_size*2,
                                        out_features=target_class,
                                        bias=False)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def rotary_positional_embedding(self, position):
        d_model = int((self.position_embedding_size - 1) / 2)
        position = position.unsqueeze(dim=2)
        freqs = torch.exp(torch.linspace(0., -1., int(d_model // 2)) * torch.log(torch.tensor(10000.))).to(self.device)
        freqs = freqs.unsqueeze(dim=0).unsqueeze(dim=0).expand((position.shape[0], 1, freqs.shape[0]))
        angles = position * freqs
        rotary_matrix = torch.stack([torch.sin(angles), torch.cos(angles)], axis=-1).to(self.device)
        return rotary_matrix.reshape((position.shape[0], position.shape[1], d_model))

    def sinusoidal_positional_encoding(self, position):
        d_model = int((self.position_embedding_size - 1) / 2)
        position = position.unsqueeze(dim=2)
        angle_rads = torch.arange(d_model) // 2 * torch.pi / torch.pow(10000, 2 * (torch.arange(d_model) // 2) / d_model)
        angle_rads = angle_rads.to(self.device)
        angle_rads = angle_rads.unsqueeze(dim=0).unsqueeze(dim=0).expand((position.shape[0], 1, angle_rads.shape[0]))
        angle_rads = torch.bmm(position, angle_rads)
        pos_encoding = torch.zeros((angle_rads.shape[0], angle_rads.shape[1], angle_rads.shape[2])).to(self.device)
        pos_encoding[:, :, 0::2] = torch.sin(angle_rads[:, :, 0::2])
        pos_encoding[:, :, 1::2] = torch.cos(angle_rads[:, :, 1::2])
        return pos_encoding

    def forward(self, x):
        word_embedding_ent1 = self.w2v(x[:, :, 0])
        tag_embedding_ent1 = self.dropout(self.relu(self.tag_embedding(x[:, :, 1].int())))

        if self.position_embedding_type == 'rotary':
            position_embedding_ent1 = x[:, :, 2:6].float()
            pos1 = self.rotary_positional_embedding(position_embedding_ent1[:, :, 0])
            pos2 = self.rotary_positional_embedding(position_embedding_ent1[:, :, 1])
            position_embedding_ent1 = torch.cat((pos1, pos2, position_embedding_ent1[:, :, 2:]), dim=2)            
        elif self.position_embedding_type == 'sinusoidal':
            position_embedding_ent1 = x[:, :, 2:6].float()
            pos1 = self.sinusoidal_positional_encoding(position_embedding_ent1[:, :, 0])
            pos2 = self.sinusoidal_positional_encoding(position_embedding_ent1[:, :, 1])
            position_embedding_ent1 = torch.cat((pos1, pos2, position_embedding_ent1[:, :, 2:]), dim=2)
        elif self.position_embedding_type == 'linear':
            position_embedding_ent1 = self.normalize_position(x[:, :, 2:6].float())
            position_embedding_ent1 = self.dropout(self.relu(position_embedding_ent1))

        direction_embedding = self.dropout(self.relu(self.direction_embedding(x[:, :, 6].int())))
        edge_embedding = self.dropout(self.relu(self.edge_embedding(x[:, :, 7].int())))

        word_embedding_ent2 = self.w2v(x[:, :, 8])
        tag_embedding_ent2 = self.dropout(self.relu(self.tag_embedding(x[:, :, 9].int())))

        if self.position_embedding_type == 'rotary':
            position_embedding_ent2 = x[:, :, 10:14].float()
            pos3 = self.rotary_positional_embedding(position_embedding_ent2[:, :, 0])
            pos4 = self.rotary_positional_embedding(position_embedding_ent2[:, :, 1])
            position_embedding_ent2 = torch.cat((pos3, pos4, position_embedding_ent2[:, :, 2:]), dim=2)    
        elif self.position_embedding_type == 'sinusoidal':
            position_embedding_ent2 = x[:, :, 10:14].float()
            pos3 = self.sinusoidal_positional_encoding(position_embedding_ent2[:, :, 0])
            pos4 = self.sinusoidal_positional_encoding(position_embedding_ent2[:, :, 1])
            position_embedding_ent2 = torch.cat((pos3, pos4, position_embedding_ent2[:, :, 2:]), dim=2) 
        elif self.position_embedding_type == 'linear':
            position_embedding_ent2 = self.normalize_position(x[:, :, 10:14].float())
            position_embedding_ent2 = self.dropout(self.relu(position_embedding_ent2))

        tokens_ent1 = torch.cat((word_embedding_ent1, tag_embedding_ent1, position_embedding_ent1), dim=2).float()
        tokens_ent2 = torch.cat((word_embedding_ent2, tag_embedding_ent2, position_embedding_ent2), dim=2).float()
        dep = torch.cat((direction_embedding, edge_embedding), dim=2).float()

        tokens_ent1 = self.dropout(self.relu(self.normalize_tokens(tokens_ent1)))
        tokens_ent2 = self.dropout(self.relu(self.normalize_tokens(tokens_ent2)))
        dep = self.dropout(self.relu(self.normalize_dep(dep)))

        x = torch.cat((tokens_ent1, dep, tokens_ent2), dim=2)

        if self.model_option == 'cnn':
            x = x.unsqueeze(1)
            
            x1 = self.conv1(x)
            x2 = self.conv2(x)
            x3 = self.conv3(x)

            x1 = torch.max(x1.squeeze(dim=3), dim=2)[0]
            x2 = torch.max(x2.squeeze(dim=3), dim=2)[0]
            x3 = torch.max(x3.squeeze(dim=3), dim=2)[0]

            x = torch.cat((x1, x2, x3), dim=1)
        elif self.model_option == 'lstm' or self.model_option == 'bilstm':
            x, _ = self.lstm(x)
            x = x[:, -1, :]

        # classifier
        if self.classifier == True:
            x = self.dense_to_tag(x)
            x = self.softmax(x)

        return x

class BertModel(nn.Module):
    def __init__(self,
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
                 classifier: bool = False,
                 model_option: str = 'cnn',
                 position_embedding_type: str = 'linear',
                 device: str = 'cpu',
                 **kwargs
                 ):
        super(BertModel, self).__init__()
        self.classifier = classifier
        self.word_embedding_size = word_embedding_size
        self.model_option = model_option
        self.position_embedding_size = position_embedding_size
        self.position_embedding_type = position_embedding_type
        self.device = device

        if 'attention_option' in kwargs.keys():
            self.attention_option = kwargs['attention_option']
        else:
            self.attention_option = False

        if self.model_option == 'lstm' or self.model_option == 'bilstm':
            self.lstm_hidden_size = kwargs['lstm_hidden_size']
            self.lstm_num_layers = kwargs['lstm_num_layers']

        self.tag_embedding = nn.Embedding(tag_number, tag_embedding_size, padding_idx=0)
        self.direction_embedding = nn.Embedding(direction_number, direction_embedding_size, padding_idx=0)
        self.edge_embedding = nn.Embedding(edge_number, edge_embedding_size, padding_idx=0)
        
        self.normalize_position = nn.Linear(in_features=position_number,
                                            out_features=position_embedding_size,
                                            bias=False)
        
        self.normalize_tokens = nn.Linear(in_features=word_embedding_size+tag_embedding_size+position_embedding_size,
                                          out_features=token_embedding_size,
                                          bias=False)
        
        self.normalize_dep = nn.Linear(in_features=direction_embedding_size+edge_embedding_size,
                                       out_features=dep_embedding_size,
                                       bias=False)
        
        self.dropout = nn.Dropout(dropout_rate)

        if self.model_option == 'cnn':
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=1,
                        out_channels=conv1_out_channels,
                        kernel_size=(conv1_length, token_embedding_size*2+dep_embedding_size),
                        stride=1,
                        bias=False),
                nn.ReLU()
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=1,
                        out_channels=conv2_out_channels,
                        kernel_size=(conv2_length, token_embedding_size*2+dep_embedding_size),
                        stride=1,
                        bias=False),
                nn.ReLU()
            )

            self.conv3 = nn.Sequential(
                nn.Conv2d(in_channels=1,
                        out_channels=conv3_out_channels,
                        kernel_size=(conv3_length, token_embedding_size*2+dep_embedding_size),
                        stride=1,
                        bias=False),
                nn.ReLU()
            )
            self.dense_to_tag = nn.Linear(in_features=conv1_out_channels + conv2_out_channels + conv3_out_channels,
                                          out_features=target_class,
                                          bias=False)
        elif self.model_option == 'lstm':
            self.lstm = nn.LSTM(input_size=token_embedding_size * 2+dep_embedding_size,
                                hidden_size=self.lstm_hidden_size,
                                num_layers=self.lstm_num_layers,
                                batch_first=True,
                                bidirectional=False,
                                dropout=dropout_rate)
            self.dense_to_tag = nn.Linear(in_features=self.lstm_hidden_size,
                                          out_features=target_class,
                                          bias=False)
        elif self.model_option == 'bilstm':
            self.lstm = nn.LSTM(input_size=token_embedding_size*2+dep_embedding_size,
                                hidden_size=self.lstm_hidden_size,
                                num_layers=self.lstm_num_layers,
                                batch_first=True,
                                bidirectional=True,
                                dropout=dropout_rate)
            self.dense_to_tag = nn.Linear(in_features=self.lstm_hidden_size,
                                          out_features=target_class,
                                          bias=False)
            
        self.self_attention=nn.MultiheadAttention(embed_dim=token_embedding_size*2+dep_embedding_size, 
                                            num_heads=4, 
                                            dropout=dropout_rate,
                                            batch_first=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def rotary_positional_embedding(self, position):
        d_model = int((self.position_embedding_size - 1) / 2)
        position = position.unsqueeze(dim=2)
        freqs = torch.exp(torch.linspace(0., -1., int(d_model // 2)) * torch.log(torch.tensor(10000.))).to(self.device)
        freqs = freqs.unsqueeze(dim=0).unsqueeze(dim=0).expand((position.shape[0], 1, freqs.shape[0]))
        angles = position * freqs
        rotary_matrix = torch.stack([torch.sin(angles), torch.cos(angles)], axis=-1).to(self.device)
        return rotary_matrix.reshape((position.shape[0], position.shape[1], d_model))

    def sinusoidal_positional_encoding(self, position):
        d_model = int((self.position_embedding_size - 1) / 2)
        position = position.unsqueeze(dim=2)
        angle_rads = torch.arange(d_model) // 2 * torch.pi / torch.pow(10000, 2 * (torch.arange(d_model) // 2) / d_model)
        angle_rads = angle_rads.to(self.device)
        angle_rads = angle_rads.unsqueeze(dim=0).unsqueeze(dim=0).expand((position.shape[0], 1, angle_rads.shape[0]))
        angle_rads = torch.bmm(position, angle_rads)
        pos_encoding = torch.zeros((angle_rads.shape[0], angle_rads.shape[1], angle_rads.shape[2])).to(self.device)
        pos_encoding[:, :, 0::2] = torch.sin(angle_rads[:, :, 0::2])
        pos_encoding[:, :, 1::2] = torch.cos(angle_rads[:, :, 1::2])
        return pos_encoding

    def forward(self, x):
        word_embedding_ent1 = x[:, :, 14:14+self.word_embedding_size]
        tag_embedding_ent1 = self.dropout(self.relu(self.tag_embedding(x[:, :, 1].int())))

        if self.position_embedding_type == 'rotary':
            position_embedding_ent1 = x[:, :, 2:6].float()
            pos1 = self.rotary_positional_embedding(position_embedding_ent1[:, :, 0])
            pos2 = self.rotary_positional_embedding(position_embedding_ent1[:, :, 1])
            position_embedding_ent1 = torch.cat((pos1, pos2, position_embedding_ent1[:, :, 2:]), dim=2)            
        elif self.position_embedding_type == 'sinusoidal':
            position_embedding_ent1 = x[:, :, 2:6].float()
            pos1 = self.sinusoidal_positional_encoding(position_embedding_ent1[:, :, 0])
            pos2 = self.sinusoidal_positional_encoding(position_embedding_ent1[:, :, 1])
            position_embedding_ent1 = torch.cat((pos1, pos2, position_embedding_ent1[:, :, 2:]), dim=2)
        elif self.position_embedding_type == 'linear':
            position_embedding_ent1 = self.normalize_position(x[:, :, 2:6].float())
            position_embedding_ent1 = self.dropout(self.relu(position_embedding_ent1))

        direction_embedding = self.dropout(self.relu(self.direction_embedding(x[:, :, 6].int())))
        edge_embedding = self.dropout(self.relu(self.edge_embedding(x[:, :, 7].int())))

        word_embedding_ent2 = x[:, :, 14+self.word_embedding_size:14+self.word_embedding_size*2]
        tag_embedding_ent2 = self.dropout(self.relu(self.tag_embedding(x[:, :, 9].int())))

        if self.position_embedding_type == 'rotary':
            position_embedding_ent2 = x[:, :, 10:14].float()
            pos3 = self.rotary_positional_embedding(position_embedding_ent2[:, :, 0])
            pos4 = self.rotary_positional_embedding(position_embedding_ent2[:, :, 1])
            position_embedding_ent2 = torch.cat((pos3, pos4, position_embedding_ent2[:, :, 2:]), dim=2)    
        elif self.position_embedding_type == 'sinusoidal':
            position_embedding_ent2 = x[:, :, 10:14].float()
            pos3 = self.sinusoidal_positional_encoding(position_embedding_ent2[:, :, 0])
            pos4 = self.sinusoidal_positional_encoding(position_embedding_ent2[:, :, 1])
            position_embedding_ent2 = torch.cat((pos3, pos4, position_embedding_ent2[:, :, 2:]), dim=2) 
        elif self.position_embedding_type == 'linear':
            position_embedding_ent2 = self.normalize_position(x[:, :, 10:14].float())
            position_embedding_ent2 = self.dropout(self.relu(position_embedding_ent2))
        
        tokens_ent1 = torch.cat((word_embedding_ent1, tag_embedding_ent1, position_embedding_ent1), dim=2).float()
        tokens_ent2 = torch.cat((word_embedding_ent2, tag_embedding_ent2, position_embedding_ent2), dim=2).float()
        
        dep = torch.cat((direction_embedding, edge_embedding), dim=2).float()
        
        tokens_ent1 = self.dropout(self.relu(self.normalize_tokens(tokens_ent1)))
        tokens_ent2 = self.dropout(self.relu(self.normalize_tokens(tokens_ent2)))
        dep = self.dropout(self.relu(self.normalize_dep(dep)))
        
        x = torch.cat((tokens_ent1, dep, tokens_ent2), dim=2)

        if self.attention_option == True:
            x, attention_weights_output = self.self_attention(x, x, x)

        if self.model_option == 'cnn':
            x = x.unsqueeze(1)
            
            x1 = self.conv1(x)
            x2 = self.conv2(x)
            x3 = self.conv3(x)

            x1 = torch.max(x1.squeeze(dim=3), dim=2)[0]
            x2 = torch.max(x2.squeeze(dim=3), dim=2)[0]
            x3 = torch.max(x3.squeeze(dim=3), dim=2)[0]

            x = torch.cat((x1, x2, x3), dim=1)
        elif self.model_option == 'lstm' or self.model_option == 'bilstm':
            x, _ = self.lstm(x)
            x = x[:, -1, :]

        if self.classifier == True:
            x = self.dense_to_tag(x)
            x = self.softmax(x)

        return x