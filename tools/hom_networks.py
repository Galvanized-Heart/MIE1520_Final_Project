import torch
import torch.nn.functional as F

from torch_geometric.nn import GraphConv, GATConv, SAGEConv, Linear, global_mean_pool

from tools.het_networks import MLP

class HomoGNN_GraphConv(torch.nn.Module):
    def __init__(self,
                 hidden_channels,
                 num_classes,
                 mlp_layers=2,
                 conv_layers=2,
                 act=F.relu,
                 aggr='mean',
                 dropout=0.5,
                 use_skip_connections=False
        ):
        
        super().__init__()
        
        self.node_emb_layers = torch.nn.ModuleList([MLP(-1, hidden_channels, hidden_channels, mlp_layers, dropout=dropout, act=act)])

        self.conv_blocks = torch.nn.ModuleList()
        for _ in range(conv_layers):
            conv = GraphConv((-1, -1), hidden_channels, aggr=aggr)
            post_lin = MLP(-1, hidden_channels, hidden_channels, mlp_layers, dropout=dropout, act=act)
            self.conv_blocks.append(torch.nn.ModuleDict({'conv': conv, 'post_lin': post_lin}))

        self.classifier = Linear(hidden_channels, num_classes)
        self.act = act
        self.dropout = torch.nn.Dropout(dropout)
        self.use_skip_connections = use_skip_connections

    def forward(self, x, edge_index, batch):

        x = self.node_emb_layers[0](x)

        for conv in self.conv_blocks:
            residual = x
            x = conv['conv'](x, edge_index)
            x = conv['post_lin'](x)
            if self.use_skip_connections:
                x = x + residual
            x = self.act(x)
            x = self.dropout(x)
        
        pooled = global_mean_pool(x, batch)
        out = self.classifier(pooled)
        return out


class HomoGNN_SAGEConv(torch.nn.Module):
    def __init__(self,
                 hidden_channels,
                 num_classes,
                 mlp_layers=2,
                 conv_layers=2,
                 act=F.relu,
                 aggr='mean',
                 dropout=0.5,
                use_skip_connections=False
        ):
        
        super().__init__()
        
        self.node_emb_layers = torch.nn.ModuleList([MLP(-1, hidden_channels, hidden_channels, mlp_layers, dropout=dropout, act=act)])

        self.conv_blocks = torch.nn.ModuleList()
        for _ in range(conv_layers):
            conv = SAGEConv((-1, -1), hidden_channels, aggr=aggr)
            post_lin = MLP(-1, hidden_channels, hidden_channels, mlp_layers, dropout=dropout, act=act)
            self.conv_blocks.append(torch.nn.ModuleDict({'conv': conv, 'post_lin': post_lin}))

        self.classifier = Linear(hidden_channels, num_classes)
        self.act = act
        self.dropout = torch.nn.Dropout(dropout)
        self.use_skip_connections = use_skip_connections

    def forward(self, x, edge_index, batch):

        x = self.node_emb_layers[0](x)

        for conv in self.conv_blocks:
            residual = x
            x = conv['conv'](x, edge_index)
            x = conv['post_lin'](x)
            if self.use_skip_connections:
                x = x + residual
            x = self.act(x)
            x = self.dropout(x)
        
        pooled = global_mean_pool(x, batch)
        out = self.classifier(pooled)
        return out
    
class HomoGNN_GATConv(torch.nn.Module):
    def __init__(self,
                 hidden_channels,
                 num_classes,
                 mlp_layers=2,
                 conv_layers=2,
                 act=F.relu,
                 aggr='mean',
                 dropout=0.5,
                use_skip_connections=False
        ):
        
        super().__init__()
        
        self.node_emb_layers = torch.nn.ModuleList([MLP(-1, hidden_channels, hidden_channels, mlp_layers, dropout=dropout, act=act)])

        self.conv_blocks = torch.nn.ModuleList()
        for _ in range(conv_layers):
            conv = GATConv((-1, -1), hidden_channels, aggr=aggr)
            post_lin = MLP(-1, hidden_channels, hidden_channels, mlp_layers, dropout=dropout, act=act)
            self.conv_blocks.append(torch.nn.ModuleDict({'conv': conv, 'post_lin': post_lin}))

        self.classifier = Linear(hidden_channels, num_classes)
        self.act = act
        self.dropout = torch.nn.Dropout(dropout)
        self.use_skip_connections = use_skip_connections

    def forward(self, x, edge_index, batch):

        x = self.node_emb_layers[0](x)

        for conv in self.conv_blocks:
            residual = x
            x = conv['conv'](x, edge_index)
            x = conv['post_lin'](x)
            if self.use_skip_connections:
                x = x + residual
            x = self.act(x)
            x = self.dropout(x)
        
        pooled = global_mean_pool(x, batch)
        out = self.classifier(pooled)
        return out