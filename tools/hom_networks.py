import torch
import torch.nn.functional as F

from torch_geometric.nn import GraphConv, GATConv, SAGEConv, Linear, global_mean_pool

from het_networks import MLP



class HomoGNN_GraphConv(torch.nn.Module):
    def __init__(self,
                 hidden_channels,
                 mlp_layers=2,
                 conv_layers=2,
                 act=F.relu,
                 aggr='mean',
                 dropout=0.5
        ):
        
        super().__init__()
        
        self.node_emb_layers = torch.nn.ModuleList(MLP(-1, hidden_channels, hidden_channels, mlp_layers, dropout=dropout, act=act))

        self.convs = torch.nn.ModuleList()
        for _ in range(conv_layers):
            self.convs.append(GraphConv((-1, -1), hidden_channels, aggr=aggr))
            

        self.classifier = Linear(hidden_channels, hidden_channels)
        self.act = act
        self.feat_dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.act(x)
            x = self.feat_dropout(x)
        
        pooled = global_mean_pool(x, batch)
        out = self.classifier(pooled)
        return out


    
class HomoSAGE(torch.nn.Module):
    def __init__(self,
                 hidden_channels,
                 num_layers=2,
                 act=F.relu,
                 aggr='mean',
                 feat_dropout=0.0
        ):
        
        super().__init__()
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv((-1, -1), hidden_channels, aggr=aggr))

        self.classifier = Linear(hidden_channels, hidden_channels)
        self.act = act
        self.feat_dropout = torch.nn.Dropout(feat_dropout)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.act(x)
            x = self.feat_dropout(x)
        
        pooled = global_mean_pool(x, batch)
        out = self.classifier(pooled)
        return out



class HomoGAT(torch.nn.Module):
    def __init__(self,
                 hidden_channels,
                 num_layers=2,
                 act=F.relu,
                 aggr='mean',
                 feat_dropout=0.0
        ):
        
        super().__init__()
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATConv((-1, -1), hidden_channels, aggr=aggr))

        self.classifier = Linear(hidden_channels, hidden_channels)
        self.act = act
        self.feat_dropout = torch.nn.Dropout(feat_dropout)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.act(x)
            x = self.feat_dropout(x)
        
        pooled = global_mean_pool(x, batch)
        out = self.classifier(pooled)
        return out