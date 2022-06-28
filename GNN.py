import torch
import GAT


class LSTM_variable_input(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.lstm1 = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm2 = torch.nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = torch.nn.Dropout(0.2)
        self.Linear = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, s):
        x = self.dropout(x)
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        out_pack, _ = self.lstm1(x_pack)
        _, (ht, _) = self.lstm2(out_pack)
        output = self.dropout(ht[-1])
        output = self.Linear(output)
        return output


class personalized_graph_network(torch.nn.Module):
    def __init__(self, in_channels, out_channels, device, dropout_rate=0.2):
        super(personalized_graph_network, self).__init__()
        self.device = device
        self.GAT = GAT.GAT(in_channels=out_channels, hidden_channels=out_channels, num_layers=1,
                           out_channels=out_channels)
        self.LSTM_module = LSTM_variable_input(embedding_dim=in_channels, hidden_dim=out_channels)
        self.User_linear = torch.nn.Linear(out_channels, out_channels)
        self.in_channels = in_channels
        self.batchnorm = torch.nn.BatchNorm1d(out_channels, affine=True)

        self.dropout = torch.nn.Dropout()

    def forward(self, graph_batch, max_length=50):
        feature_transformed = graph_batch.x.to(self.device)
        feature_map = graph_batch.edge_index.to(self.device)
        user_position = graph_batch.user_index.tolist()
        ptr = graph_batch.ptr.tolist()
        LSTM_feature = torch.empty(size=(len(user_position), max_length, 768)).to(self.device)
        final_user_position = torch.tensor([i + ptr[index] for index, i in enumerate(user_position)]).to(self.device)
        temp = 0
        length_all = []
        for flag, index in enumerate(ptr[1:]):
            current_user_history = feature_transformed[temp:index - 1]
            if len(current_user_history) < max_length:
                length_all.append(len(current_user_history))
                zero_padding = torch.zeros((max_length) - len(current_user_history), 768)
                LSTM_feature[flag] = torch.cat((zero_padding, current_user_history), dim=0).to(self.device)
            elif len(current_user_history) > max_length:
                length_all.append(max_length)
                LSTM_feature[flag] = current_user_history[-max_length:]
            else:
                length_all.append(max_length)
                LSTM_feature[flag] = current_user_history
            temp = index

        LSTM_output = self.batchnorm(self.LSTM_module(LSTM_feature, length_all))
        GNN_output = self.batchnorm(self.GAT(x=feature_transformed, edge_index=feature_map))
        GNN_user = torch.index_select(GNN_output, 0, final_user_position)
        final_user_embedding = torch.add(LSTM_output, GNN_user)
        final_user_embedding = self.dropout(final_user_embedding)
        final_user_embedding = self.User_linear(final_user_embedding)
        return final_user_embedding
