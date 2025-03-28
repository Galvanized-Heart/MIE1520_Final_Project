import torch
from torch_geometric.data import HeteroData



def _get_node_type_mapping(hom_data, onehot_indices, expected_types):
    """
    Identify node types using one-hot columns and map them to string labels.
    
    Args:
        hom_data: Homogeneous graph data with node features.
        onehot_indices: Column indices for one-hot type encoding (e.g., [0, 1, 2]).
        expected_types: List of string labels (e.g., ["A", "B", "C"]).
    
    Returns:
        Dictionary mapping string labels to node indices (e.g., {"A": [0, 1], "B": [2]}).
    """
    type_dict = {}
    onehot_features = hom_data.x[:, onehot_indices]
    
    for node_idx, node in enumerate(onehot_features):
        # Get onehot encoded index
        type_idx = int(torch.argmax(node)) 
        if type_idx >= len(expected_types):
            raise ValueError(f"One-hot index {type_idx} is out of bounds for expected_types {expected_types}")
        
        # Map onehot indicies to expected types (e.g. 0 -> "A")
        type_label = expected_types[type_idx]
        if type_label not in type_dict:
            type_dict[type_label] = []
        type_dict[type_label].append(node_idx)
    
    return type_dict



def _create_mapping_dict(type_dict):
    """
    Create ID mapping dictionaries for all node types.
    
    This resets node indices for each node type in the heterogeneous graph.
    """
    return {t: {orig_idx: new_idx for new_idx, orig_idx in enumerate(nodes)}
            for t, nodes in type_dict.items()}

def _filter_and_remap_edges(hom_data, src_type_nodes, dst_type_nodes, 
                           src_map, dst_map, enforce_canonical):
    """
    Filter edges from hom_data.edge_index where source is in src_type_nodes and
    destination is in dst_type_nodes, and remap original node indices to new indices.
    
    If enforce_canonical is True, reorders each edge (min first) and removes duplicates.
    """
    src_tensor = torch.tensor(src_type_nodes)
    dst_tensor = torch.tensor(dst_type_nodes)
    
    # Create mask over homogeneous edge_index
    mask = torch.isin(hom_data.edge_index[0], src_tensor) & torch.isin(hom_data.edge_index[1], dst_tensor)
    filtered_edges = hom_data.edge_index[:, mask]
    
    if filtered_edges.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    
    src_indices = torch.tensor([src_map[orig.item()] for orig in filtered_edges[0]])
    dst_indices = torch.tensor([dst_map[orig.item()] for orig in filtered_edges[1]])
    edge_index = torch.stack([src_indices, dst_indices])
    
    if enforce_canonical:
        # Reorder each edge so that the smaller index is first.
        reordered_src = torch.min(edge_index[0], edge_index[1])
        reordered_dst = torch.max(edge_index[0], edge_index[1])
        edge_index = torch.stack([reordered_src, reordered_dst])
        # Remove duplicate edges
        edge_index = torch.unique(edge_index, dim=1)
    
    return edge_index



def convert_hom_to_het(
        hom_data, 
        onehot_indices=[0, 1, 2], 
        expected_types=["A", "B", "C"], 
        is_directed=True, 
        enforce_canonical=False, 
        include_meta=False):
    """
    Convert a homogeneous graph (with node features that include one-hot splitting columns)
    to a heterogeneous graph.
    
    Parameters:
      - expected_types: list of possible node types (e.g., [0, 1, 2])
      - is_directed: if True, treat the graph as directed; if False, as undirected.
      - enforce_canonical: used when is_directed is False; reorders edges and removes duplicates.
      - onehot_indices: list of column indices in hom_data.x used for one-hot splitting.
      
    All columns not in onehot_indices are treated as additional node features.
    If no additional columns exist, a tensor of ones is used.
    """
    het_data = HeteroData()
    
    # Determine node types based on the one-hot splitting columns.
    type_dict = _get_node_type_mapping(hom_data, onehot_indices, expected_types)
    mapping_dict = _create_mapping_dict(type_dict)
    
    # Determine additional feature indices (all columns not in onehot_indices).
    total_dim = hom_data.x.shape[1]
    additional_indices = sorted(list(set(range(total_dim)) - set(onehot_indices)))
    
    # Add node features for each expected type.
    for t in expected_types:
        nodes = type_dict.get(t, [])
        if nodes:
            if additional_indices:
                # Use the additional features.
                het_data[str(t)].x = hom_data.x[torch.tensor(nodes, dtype=torch.long)][:, additional_indices]
            else:
                # Default to a tensor of ones.
                het_data[str(t)].x = torch.ones((len(nodes), 1), dtype=hom_data.x.dtype)
        else:
            # No nodes of this type, so add an empty tensor with appropriate feature dim.
            feat_dim = len(additional_indices) if additional_indices else 1
            het_data[str(t)].x = torch.empty((0, feat_dim), dtype=hom_data.x.dtype)
    
    # Process edges
    if is_directed:
        type_pairs = [(src, dst) for src in expected_types for dst in expected_types]
    else:
        type_pairs = [(src, dst) for i, src in enumerate(expected_types) for dst in expected_types[i:]]
    
    for src_type, dst_type in type_pairs:
        edge_type = (str(src_type), "connects", str(dst_type))
        src_nodes = type_dict.get(src_type, [])
        dst_nodes = type_dict.get(dst_type, [])
        
        edge_index = _filter_and_remap_edges(
            hom_data,
            src_nodes,
            dst_nodes,
            mapping_dict.get(src_type, {}),
            mapping_dict.get(dst_type, {}),
            enforce_canonical=enforce_canonical
        )
        het_data[edge_type].edge_index = edge_index
    
    # Preserve graph-level label if it exists
    if hasattr(hom_data, 'y'):
        het_data.y = hom_data.y
    
    return het_data



# This could be improved by having a function to create metadata=[node_types, edge_types]
# for the entire dataset and maybe also not have empty tensors if we have metadata.
# node_types = ["A", "B", "C"]
# edge_types = [('A', 'connects', 'A'),
               #('A', 'connects', 'B'),
               #('A', 'connects', 'C'),
               #('B', 'connects', 'B'),
               #('B', 'connects', 'C'),
               #('C', 'connects', 'C')]

# Could be as simple as:
# edge_types = het_dataset[0].edge_types
# node_types = het_dataset[0].node_types
# metadata = [node_types, edge_types]