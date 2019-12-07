import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils


class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(GNN, self).__init__()

        self.convs = nn.ModuleList()
        conv_model = self.build_conv_model(args.model_type)
        self.convs.append(conv_model(input_dim, hidden_dim))
        assert (args.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(args.num_layers-1):
            self.convs.append(conv_model(hidden_dim, hidden_dim))
        
        self.dropout_layer = nn.Dropout(args.dropout)

        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.Dropout(args.dropout), 
            nn.Linear(hidden_dim, output_dim))


    def build_conv_model(self, model_type):
        if model_type == 'GCN':
            return pyg_nn.GCNConv
        elif model_type == 'GraphSage':
            return pyg_nn.SAGEConv
        elif model_type == 'GAT':
            return pyg_nn.GATConv


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = nn.ReLU()(x)
            x = self.dropout_layer(x)

        x = self.post_mp(x)

        return F.log_softmax(x, dim=1)


    def loss(self, pred, label):
        return F.nll_loss(pred, label)
