# Heterogeneous Graph Neural Networks for Protein Structure Analysis

This repository contains implementations of various graph neural network (GNN) architectures for analyzing protein structures, with a focus on both homogeneous and heterogeneous graph representations.

## Overview

The codebase provides tools for:
- Working with protein structure data from PDB files
- Converting between homogeneous and heterogeneous graph representations
- Implementing different GNN architectures (GraphConv, SAGEConv, GATConv)
- Training, evaluating, and visualizing model performance

## Installation

Clone the repository and install the dependencies:

```bash
# Clone the repository
git clone https://github.com/Galvanized-Heart/MIE1520_Final_Project.git
cd MIE1520_Final_Project

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
├── tools/
│   ├── het_networks.py    # Heterogeneous GNN implementations
│   ├── hom_networks.py    # Homogeneous GNN implementations
│   ├── pscdb_dataset.py   # Protein structure dataset handling
│   └── utils.py           # Utility functions for training, evaluation, and visualization
│
├── notebooks/             # Example notebooks demonstrating usage
├── data/                  # Directory for datasets (created during execution)
├── results/               # Directory for experiment results (created during execution)
└── requirements.txt       # Project dependencies
```

## Key Components

### Graph Neural Network Models

The repository includes several GNN architectures:

1. **Homogeneous GNNs** (`tools/hom_networks.py`):
   - `HomoGNN_GraphConv`
   - `HomoGNN_SAGEConv`
   - `HomoGNN_GATConv`

2. **Heterogeneous GNNs** (`tools/het_networks.py`):
   - `HeteroGNN_GraphConv`
   - `HeteroGNN_SAGEConv`
   - `HeteroGNN_GATConv`

### Utilities

The `utils.py` file includes essential functions for:

- Converting homogeneous graphs to heterogeneous graphs
- Training and testing models
- Visualizing metrics (loss, accuracy, F1 score)
- Ensuring reproducibility with seed control

### Dataset Handling

The `pscdb_dataset.py` file contains the `ProteinPairGraphBuilder` class for:
- Loading protein structures from PDB files
- Creating graph representations based on spatial proximity
- Aligning residues between bound and unbound protein states

## Usage

### Basic Example

```python
import torch
from torch_geometric.loader import DataLoader
from tools.hom_networks import HomoGNN_GraphConv
from tools.utils import train, test, hom_predict, reset_seeds

# Set device and ensure reproducibility
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
reset_seeds(42, device)

# Prepare your dataset and create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Initialize the model
model = HomoGNN_GraphConv(
    hidden_channels=64,
    num_classes=3,
    dropout=0.5
).to(device)

# Set up training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(100):
    train_loss, train_acc, train_f1 = train(model, train_loader, optimizer, 
                                           criterion, hom_predict, device=device)
    val_loss, val_acc, val_f1 = test(model, val_loader, criterion, 
                                     hom_predict, device=device)
    print(f"Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
```

### Converting Homogeneous to Heterogeneous Graphs

```python
from tools.utils import convert_hom_to_het

# Assuming you have a homogeneous graph in PyG Data format
hetero_graph = convert_hom_to_het(
    hom_data=homogeneous_graph,
    onehot_indices=[0, 1, 2],  # Indices of one-hot encoded node type features
    expected_node_types=["hydrophobic", "hydrophilic", "charged"],  # Your node types
    expected_edge_types=['edge_index'],  # Edge types in your data
    is_directed=True
)
```

## Extending the Repository

### Creating Your Own Experiments

You can create your own experiments by:

1. **Building custom datasets**:
   - Adapt the `ProteinPairGraphBuilder` for your PDB data
   - Or create your own graph builders for different molecular representations

2. **Implementing new GNN architectures**:
   - Extend the existing models in `het_networks.py` or `hom_networks.py`
   - Experiment with different layer types, activation functions, or aggregation methods

3. **Exploring different training strategies**:
   - Try different optimizers, learning rate schedulers
   - Implement cross-validation
   - Experiment with different loss functions for specific tasks

### Creating Your Own Heterogeneous Graph Datasets

To create your own heterogeneous graph datasets:

1. Define node types and their features
2. Define edge types and their connections
3. Use the `convert_hom_to_het` function if starting from homogeneous data
4. Or directly create `HeteroData` objects from PyTorch Geometric

Example:
```python
from torch_geometric.data import HeteroData

def create_custom_hetero_graph(data):
    het_data = HeteroData()
    
    # Add node features for each type
    het_data['hydrophobic'].x = hydrophobic_features
    het_data['hydrophilic'].x = hydrophilic_features
    
    # Add edge connections
    het_data['hydrophobic', 'connects', 'hydrophilic'].edge_index = hydrophobic_hydrophilic_edges
    
    return het_data
```

## Additional Resources

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [Protein Data Bank (PDB)](https://www.rcsb.org/)
- [PSCDB Dataset](https://github.com/a-r-j/graphein/tree/master/datasets/pscdb)

## License

[MIT License](LICENSE)
