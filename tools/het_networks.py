import torch
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv, GraphConv, GATConv, SAGEConv, Linear, global_mean_pool



class HeteroCONV(torch.nn.Module):
    def __init__(
            self,
            metadata, 
            hidden_channels,
            out_channels,
            num_layers=2,
            act=F.relu,
            intra_aggr='mean',
            inter_aggr='mean',
            feat_dropout=0.0
        ):
        
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {edge_type: GraphConv((-1, -1), hidden_channels, aggr=intra_aggr) for edge_type in metadata[1]}
            self.convs.append(HeteroConv(conv_dict, aggr=inter_aggr))

        self.classifier = Linear(len(metadata[0]) * hidden_channels, out_channels)
        self.act = act
        self.feat_dropout = torch.nn.Dropout(feat_dropout)
        
    def forward(self, x_dict, edge_index_dict, batch_dict, batch_size):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: self.feat_dropout(self.act(x)) for key, x in x_dict.items()}

        pooled_list = [global_mean_pool(x, batch_dict[node_type], size=batch_size) for node_type, x in x_dict.items()]

        graph_emb = torch.cat(pooled_list, dim=1)
        out = self.classifier(graph_emb)
        return out
    


class HeteroSAGE(torch.nn.Module):
    def __init__(
            self,
            metadata, 
            hidden_channels,
            out_channels,
            num_layers=2,
            act=F.relu,
            intra_aggr='mean',
            inter_aggr='mean',
            feat_dropout=0.0
        ):
        
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {edge_type: SAGEConv((-1, -1), hidden_channels, aggr=intra_aggr) for edge_type in metadata[1]}
            self.convs.append(HeteroConv(conv_dict, aggr=inter_aggr))

        self.classifier = Linear(len(metadata[0]) * hidden_channels, out_channels)
        self.act = act
        self.feat_dropout = torch.nn.Dropout(feat_dropout)

    def forward(self, x_dict, edge_index_dict, batch_dict, batch_size):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: self.feat_dropout(self.act(x)) for key, x in x_dict.items()}

        pooled_list = [global_mean_pool(x, batch_dict[node_type], size=batch_size) for node_type, x in x_dict.items()]

        graph_emb = torch.cat(pooled_list, dim=1)
        out = self.classifier(graph_emb)
        return out



class HeteroGAT(torch.nn.Module):
    def __init__(
            self,
            metadata, 
            hidden_channels,
            out_channels,
            num_layers=2,
            act=F.relu,
            intra_aggr='mean',
            inter_aggr='mean',
            attn_dropout=0.0,
            feat_dropout=0.0
        ): 

        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {edge_type: GATConv((-1, -1), hidden_channels, aggr=intra_aggr, add_self_loops=False, dropout=attn_dropout) for edge_type in metadata[1]}
            self.convs.append(HeteroConv(conv_dict, aggr=inter_aggr))

        self.classifier = Linear(len(metadata[0]) * hidden_channels, out_channels)
        self.act = act
        self.feat_dropout = torch.nn.Dropout(feat_dropout)

    def forward(self, x_dict, edge_index_dict, batch_dict, batch_size):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: self.feat_dropout(self.act(x)) for key, x in x_dict.items()}

        pooled_list = [global_mean_pool(x, batch_dict[node_type], size=batch_size) for node_type, x in x_dict.items()]

        graph_emb = torch.cat(pooled_list, dim=1)
        out = self.classifier(graph_emb)
        return out