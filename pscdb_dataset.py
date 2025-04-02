import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from Bio.PDB import PDBParser, PDBList

# Configuration
THRESHOLD = 5.0                         # Edge distance threshold
DATA_PATH = 'data/PSCDB/PDB_structures' # Storage location for PDB structures




class ProteinPairGraphBuilder:
    """
    Builds a PyG graph using PDB codes for a bound protein, unbound 
    protein, distance threshold in Angstroms, and motion label. 

    This class was specifically designed to extract data from PSCDB
    which can be found in 'structural_rearrangement_data.csv' at 
    https://github.com/a-r-j/graphein/tree/master/datasets/pscdb
    """
    def __init__(self, threshold=THRESHOLD, data_path=DATA_PATH):
        self.parser = PDBParser(QUIET=True)
        self.pdblist = PDBList()
        self.threshold = threshold
        self.data_path = data_path
        self.three_to_one = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
                             'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
                             'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
                             'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
                             'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
        self.aa_to_idx = {aa: i for i, aa in enumerate(sorted(self.three_to_one.values()))}
        
    def _get_residues(self, pdb_code):
        """Fetch PDB and return aligned residues with CA coordinates"""
        pdb_file = self.pdblist.retrieve_pdb_file(pdb_code, pdir=self.data_path, file_format='pdb')
        structure = self.parser.get_structure(pdb_code, pdb_file)
        residues = []

        try:
            model = next(structure.get_models())
            for chain in model:
                for residue in chain:
                    if residue.id[0] == ' ' and 'CA' in residue:
                        res_name = residue.resname
                        # Only keep standard amino acids
                        if res_name in self.three_to_one:
                            res_info = {
                                'chain': chain.id,
                                'resnum': residue.id[1],
                                'aa': res_name,  # Store 3-letter code
                                'ca': residue['CA'].coord
                            }
                            residues.append(res_info)
                return sorted(residues, key=lambda x: (x['chain'], x['resnum']))
        except Exception as e:
            print(f"Error processing {pdb_code}: {e}")
        return None

    def _create_edges(self, coords):
        """Create edges based on spatial proximity"""
        dist_matrix = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
        i, j = np.where((dist_matrix < self.threshold) & (np.eye(len(coords)) == 0))
        mask = i < j
        return torch.tensor(np.stack([i[mask], j[mask]]), dtype=torch.long)

    def _create_seq_edges(self, n_residues):
        """Create backbone sequence edges"""
        return torch.tensor([np.arange(n_residues-1), np.arange(1, n_residues)], dtype=torch.long)
    
    def _align_residues(self, free_res, bound_res):
        """Align residues present in BOTH structures"""
        free_dict = {(r['chain'], r['resnum']): r for r in free_res}
        bound_dict = {(r['chain'], r['resnum']): r for r in bound_res}
        
        # Get only residues existing in both
        common_keys = set(free_dict.keys()).intersection(bound_dict.keys())
        return [
            (free_dict[key], bound_dict[key])
            for key in sorted(common_keys)
        ]

    def build_graph_pair(self, free_pdb, bound_pdb, motion_label):
        # Get residues from both states
        free_res = self._get_residues(free_pdb)
        bound_res = self._get_residues(bound_pdb)
        
        if not free_res or not bound_res:
            return None
            
        # Align to common residues only
        aligned_pairs = self._align_residues(free_res, bound_res)
        
        if not aligned_pairs:
            print(f"No common residues between {free_pdb} and {bound_pdb}")
            return None
            
        # Report alignment statistics
        orig_free = len(free_res)
        orig_bound = len(bound_res)
        common = len(aligned_pairs)
        print(f"Aligned {common} residues (Free: {orig_free}, Bound: {orig_bound}, "
              f"Excluded: {orig_free + orig_bound - 2*common})")

        # Node features (now guaranteed to have both states)
        node_features = []
        free_coords = []
        bound_coords = []
        
        for free_item, bound_item in aligned_pairs:
            # Convert to 1-letter code
            one_letter = self.three_to_one[free_item['aa']]
            idx = self.aa_to_idx[one_letter]
            
            # One-hot encoding
            one_hot = torch.zeros(len(self.aa_to_idx))
            one_hot[idx] = 1.0

            # Coordinates and displacement
            free_ca = torch.tensor(free_item['ca'], dtype=torch.float)
            bound_ca = torch.tensor(bound_item['ca'], dtype=torch.float)
            displacement = bound_ca - free_ca
            
            features = torch.cat([
                one_hot,
                free_ca,
                bound_ca,
                displacement
            ])
            
            node_features.append(features)
            free_coords.append(free_ca)
            bound_coords.append(bound_ca)

        # Create edges based on aligned free structure
        edge_index_free = self._create_edges(torch.stack(free_coords))
        edge_index_bound = self._create_edges(torch.stack(bound_coords))
        edge_index_union = torch.unique(
            torch.cat([edge_index_free, edge_index_bound], dim=1), 
            dim=1
        )

        return Data(
            x=torch.stack(node_features),
            edge_index_free=edge_index_free,
            edge_index_bound=edge_index_bound,
            edge_index_union=edge_index_union,
            pos_free=torch.stack(free_coords),
            pos_bound=torch.stack(bound_coords),
            y=torch.tensor([motion_label], dtype=torch.long)
        )