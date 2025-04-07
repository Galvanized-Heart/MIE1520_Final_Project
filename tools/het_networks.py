import torch
import torch.nn.functional as F
from torch.nn import ModuleDict, ModuleList, Linear, Dropout, LayerNorm
from torch_geometric.nn import HeteroConv, GraphConv, GATConv, SAGEConv, Linear, global_mean_pool

class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2,
                 dropout=0.5, act=F.relu):
        super().__init__()
        self.layers = ModuleList()
        self.act = act

        self.layers.append(Linear(in_dim, hidden_dim))
        self.layers.append(Dropout(dropout))

        for _ in range(num_layers-2):
            self.layers.append(Linear(hidden_dim, hidden_dim))
            self.layers.append(Dropout(dropout))

        self.layers.append(Linear(hidden_dim, out_dim))

    def forward(self, x):
        for layer in self.layers:
            x = self.act(layer(x))
        return x

# --------------------------- #

class HeteroGNN_GraphConv(torch.nn.Module):
    def __init__(
            self,
            metadata, 
            hidden_channels,
            num_classes,
            mlp_layers=2,
            conv_layers=2,
            act=F.relu,
            intra_aggr='mean',
            inter_aggr='sum',
            dropout=0.5,
            use_skip_connections=False
        ):

        super().__init__()

        node_types, edge_types = metadata
        self.node_types = node_types

        self.node_emb_layers = ModuleDict({nt: MLP(-1, hidden_channels, hidden_channels, mlp_layers, dropout=dropout, act=act) for nt in node_types})

        self.conv_blocks = ModuleList()
        for _ in range(conv_layers):
            conv_dict = {et: GraphConv((-1, -1), hidden_channels, aggr=intra_aggr) for et in edge_types}
            hetero_conv = HeteroConv(conv_dict, aggr=inter_aggr)
            post_lin = ModuleDict({nt: MLP(-1, hidden_channels, hidden_channels, mlp_layers, dropout=dropout, act=act) for nt in node_types})
            self.conv_blocks.append(ModuleDict({'conv': hetero_conv, 'post_lin': post_lin}))

        self.classifier = Linear(len(node_types) * hidden_channels, num_classes)
        self.act = act
        self.dropout = Dropout(dropout)
        self.use_skip_connections = use_skip_connections
        
    def forward(self, x_dict, edge_index_dict, batch_dict, batch_size):

        x_dict = {nt: self.node_emb_layers[nt](x) for nt, x in x_dict.items()}

        for block in self.conv_blocks:
            residual_dict = x_dict
            x_dict = block['conv'](x_dict, edge_index_dict)
            x_dict = {nt: block['post_lin'][nt](x_dict[nt]) for nt in x_dict}
            if self.use_skip_connections:
                x_dict = {nt: x + residual_dict[nt] for nt, x in x_dict.items()}
            x_dict = {nt: self.act(x) for nt, x in x_dict.items()}
            x_dict = {nt: self.dropout(x) for nt, x in x_dict.items()}

        pooled_list = [global_mean_pool(x, batch_dict[node_type], size=batch_size) for node_type, x in x_dict.items()]

        graph_emb = torch.cat(pooled_list, dim=1)
        out = self.classifier(graph_emb)
        return out
    
# --------------------------- #

class HeteroGNN_SAGEConv(torch.nn.Module):
    def __init__(
            self,
            metadata, 
            hidden_channels,
            num_classes,
            mlp_layers=2,
            conv_layers=2,
            act=F.relu,
            intra_aggr='sum',
            inter_aggr='mean',
            dropout=0.5
        ):

        super().__init__()

        node_types, edge_types = metadata
        self.node_types = node_types

        self.node_emb_layers = ModuleDict({nt: MLP(-1, hidden_channels, hidden_channels, mlp_layers, dropout=dropout, act=act) for nt in node_types})

        self.conv_blocks = ModuleList()
        for _ in range(conv_layers):
            conv_dict = {et: SAGEConv((-1, -1), hidden_channels, aggr=intra_aggr) for et in edge_types}
            hetero_conv = HeteroConv(conv_dict, aggr=inter_aggr)
            post_lin = ModuleDict({nt: MLP(-1, hidden_channels, hidden_channels, mlp_layers, dropout=dropout, act=act) for nt in node_types})
            self.conv_blocks.append(ModuleDict({'conv': hetero_conv, 'post_lin': post_lin}))

        self.classifier = Linear(len(node_types) * hidden_channels, num_classes)
        self.act = act
        self.dropout = Dropout(dropout)
        
    def forward(self, x_dict, edge_index_dict, batch_dict, batch_size):

        x_dict = {nt: self.node_emb_layers[nt](x) for nt, x in x_dict.items()}

        for block in self.conv_blocks:
            x_dict = block['conv'](x_dict, edge_index_dict)
            x_dict = {nt: self.dropout(self.act(block['post_lin'][nt](x))) for nt, x in x_dict.items()}

        pooled_list = [global_mean_pool(x, batch_dict[node_type], size=batch_size) for node_type, x in x_dict.items()]

        graph_emb = torch.cat(pooled_list, dim=1)
        out = self.classifier(graph_emb)
        return out
    
# --------------------------- #

class HeteroGNN_GATConv(torch.nn.Module):
    def __init__(
            self,
            metadata, 
            hidden_channels,
            num_classes,
            mlp_layers=2,
            conv_layers=2,
            act=F.relu,
            intra_aggr='sum',
            inter_aggr='mean',
            dropout=0.5
        ):

        super().__init__()

        node_types, edge_types = metadata
        self.node_types = node_types

        self.node_emb_layers = ModuleDict({nt: MLP(-1, hidden_channels, hidden_channels, mlp_layers, dropout=dropout, act=act) for nt in node_types})

        self.conv_blocks = ModuleList()
        for _ in range(conv_layers):
            conv_dict = {et: GATConv((-1, -1), hidden_channels, aggr=intra_aggr, add_self_loops=False) for et in edge_types}
            hetero_conv = HeteroConv(conv_dict, aggr=inter_aggr)
            post_lin = ModuleDict({nt: MLP(-1, hidden_channels, hidden_channels, mlp_layers, dropout=dropout, act=act) for nt in node_types})
            self.conv_blocks.append(ModuleDict({'conv': hetero_conv, 'post_lin': post_lin}))

        self.classifier = Linear(len(node_types) * hidden_channels, num_classes)
        self.act = act
        self.dropout = Dropout(dropout)
        

    def forward(self, x_dict, edge_index_dict, batch_dict, batch_size):

        x_dict = {nt: self.node_emb_layers[nt](x) for nt, x in x_dict.items()}

        for block in self.conv_blocks:
            x_dict = block['conv'](x_dict, edge_index_dict)
            x_dict = {nt: self.dropout(self.act(block['post_lin'][nt](x))) for nt, x in x_dict.items()}

        pooled_list = [global_mean_pool(x, batch_dict[node_type], size=batch_size) for node_type, x in x_dict.items()]

        graph_emb = torch.cat(pooled_list, dim=1)
        out = self.classifier(graph_emb)
        return out