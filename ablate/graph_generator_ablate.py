"""
Constructs molecular graphs with 19 node features mapped to a 29D space, 7 bond attributes mapped to 7D space. Laplacian positional encodings (PE) will be computed for each node in the 
spectra graph. PE has a dimension of 9.

Generate pytorch geometry Data objects for molecules. Nodes, Edges and PE in Data object.
The program automatically saves to your output directory, which is prefered be same as input directory. Data object(s) is/are a list(s)
"""

import dgl
import rdkit
import torch
import logging
import numpy as np
from rdkit import Chem, RDLogger
import networkx as nx
from tqdm import tqdm
import graphein.molecule as gm
from functools import partial
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalFeatures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
RDLogger.DisableLog('rdApp.*')

hybridization_map = {
    rdkit.Chem.rdchem.HybridizationType.S: 1,
    rdkit.Chem.rdchem.HybridizationType.SP: 2,
    rdkit.Chem.rdchem.HybridizationType.SP2: 3,
    rdkit.Chem.rdchem.HybridizationType.SP3: 4,
    rdkit.Chem.rdchem.HybridizationType.SP3D: 5,
    rdkit.Chem.rdchem.HybridizationType.SP3D2: 6,
    rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED: 7,
    rdkit.Chem.rdchem.HybridizationType.OTHER: 0
}

chiral_tag_map = {
    rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 1,
    rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 2,
    rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 3,
    rdkit.Chem.rdchem.ChiralType.CHI_SQUAREPLANAR: 4,
    rdkit.Chem.rdchem.ChiralType.CHI_TRIGONALBIPYRAMIDAL: 5,
    rdkit.Chem.rdchem.ChiralType.CHI_OCTAHEDRAL: 6,
    rdkit.Chem.rdchem.ChiralType.CHI_OTHER: 0
}

bond_type_map = {
    rdkit.Chem.rdchem.BondType.UNSPECIFIED: 1,
    rdkit.Chem.rdchem.BondType.SINGLE: 2,
    rdkit.Chem.rdchem.BondType.DOUBLE: 3,
    rdkit.Chem.rdchem.BondType.TRIPLE: 4,
    rdkit.Chem.rdchem.BondType.QUADRUPLE:5,
    rdkit.Chem.rdchem.BondType.QUINTUPLE: 6,
    rdkit.Chem.rdchem.BondType.HEXTUPLE: 7,
    rdkit.Chem.rdchem.BondType.ONEANDAHALF: 8,
    rdkit.Chem.rdchem.BondType.TWOANDAHALF: 9,
    rdkit.Chem.rdchem.BondType.THREEANDAHALF: 10,
    rdkit.Chem.rdchem.BondType.FOURANDAHALF: 11,
    rdkit.Chem.rdchem.BondType.FIVEANDAHALF: 12,
    rdkit.Chem.rdchem.BondType.AROMATIC: 13,
    rdkit.Chem.rdchem.BondType.IONIC: 14,
    rdkit.Chem.rdchem.BondType.HYDROGEN: 15,
    rdkit.Chem.rdchem.BondType.THREECENTER: 16,
    rdkit.Chem.rdchem.BondType.DATIVEONE: 17,
    rdkit.Chem.rdchem.BondType.DATIVE: 18,
    rdkit.Chem.rdchem.BondType.DATIVEL: 19,
    rdkit.Chem.rdchem.BondType.DATIVER: 20,
    rdkit.Chem.rdchem.BondType.ZERO: 0
}

bond_stereo_map = {
    rdkit.Chem.rdchem.BondStereo.STEREONONE: 0,
    rdkit.Chem.rdchem.BondStereo.STEREOANY: 1,
    rdkit.Chem.rdchem.BondStereo.STEREOZ: 2,
    rdkit.Chem.rdchem.BondStereo.STEREOE: 3,
    rdkit.Chem.rdchem.BondStereo.STEREOCIS: 4,
    rdkit.Chem.rdchem.BondStereo.STEREOTRANS: 5,
    rdkit.Chem.rdchem.BondStereo.STEREOATROPCW: 6,
    rdkit.Chem.rdchem.BondStereo.STEREOATROPCCW: 7
}

def encode_hybridization(hybridization):
    return int(hybridization_map.get(hybridization, 7))

def encode_chiral_tag(chiral_tag):
    return int(chiral_tag_map.get(chiral_tag, 0))

def encode_bond_type(bond_type):
    return int(bond_type_map.get(bond_type, 0))

def encode_bond_stereo(stereo_type):
    return int(bond_stereo_map.get(stereo_type, 0))

def encode_boolean(value):
    return 1 if value else 0

def encode_graph_features(node_features, edge_features):
    encoded_node_features = []
    for node_feats in node_features:
        encoded = []
        for idx, feat in enumerate(node_feats):
            if idx == 12:
                encoded.append(encode_hybridization(feat))
            elif idx == 15:
                encoded.append(encode_chiral_tag(feat))
            elif idx in [13, 14, 16, 17, 18]:
                encoded.append(encode_boolean(feat))
            else:
                encoded.append(feat)
        encoded_node_features.append(encoded)

    encoded_edge_features = []
    for edge_feats in edge_features:
        encoded = []
        for idx, feat in enumerate(edge_feats):
            if idx == 0:
                encoded.append(encode_bond_type(feat))
            elif idx == 4:
                encoded.append(encode_bond_stereo(feat))
            elif idx in [1, 2, 3, 5, 6]:
                encoded.append(encode_boolean(feat))
            else:
                encoded.append(feat)
        encoded_edge_features.append(encoded)

    return encoded_node_features, encoded_edge_features

def run_graph_generator(smiles_list):
   
    config = gm.MoleculeGraphConfig(
            node_metadata_functions=[
            gm.atom_type_one_hot,
            gm.atomic_mass,
            gm.degree,
            gm.total_degree,
            gm.total_valence,
            gm.explicit_valence,
            gm.implicit_valence,
            gm.num_explicit_h,
            gm.num_implicit_h,
            gm.total_num_h,
            gm.num_radical_electrons,
            gm.formal_charge,
            gm.hybridization,
            gm.is_aromatic,
            gm.is_isotope,
            gm.chiral_tag,
            gm.is_ring,
            partial(gm.is_ring_size, ring_size=5),
            partial(gm.is_ring_size, ring_size=7)
        ],

        edge_metadata_functions=[
            gm.add_bond_type,
            gm.bond_is_aromatic,
            gm.bond_is_conjugated,
            gm.bond_is_in_ring,
            gm.bond_stereo,
            partial(gm.bond_is_in_ring_size, ring_size=5),
            partial(gm.bond_is_in_ring_size, ring_size=7)
        ]

    )
   
    node_features_tensor_list = []
    edge_features_tensor_list = []
    edge_pairs_list = []
    laplacian_positional_embed_9_dim = []

    for smiles in tqdm(smiles_list, desc="Computing graph and extracting data...", leave=False):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.error(f"Invalid SMILES string: {smiles}")
                continue
            graph = gm.construct_graph(smiles=smiles, config=config)
        except Exception as e:
            logger.error(f"Error constructing graph for SMILES {smiles}: {e}")
            continue

        node_features = []
        edge_features = []
        edge_pairs_num = []
        node_ids_labels = []

        node_id_mapping = {nid: int(nid.split(':')[1]) for nid in graph.nodes}

        for n, d in graph.nodes(data=True):
            all_node_values = list(d.values())
            node_features.append(all_node_values[4:] if len(all_node_values) > 4 else [])
            node_ids_labels.append(n.split(':')[0])

        for u, v, d in graph.edges(data=True):
            all_edge_values = list(d.values())
            edge_features.append(all_edge_values[2:] if len(all_edge_values) > 2 else [])
            edge_pairs_num.append((node_id_mapping[u], node_id_mapping[v]))

        node_features, edge_features = encode_graph_features(node_features, edge_features)

        num_nodes = len(node_ids_labels)
        src, dst = zip(*edge_pairs_num) if edge_pairs_num else ([], [])
        src = torch.tensor(src, dtype=torch.int64)
        dst = torch.tensor(dst, dtype=torch.int64)
        g = dgl.graph((src, dst), num_nodes=num_nodes)

        if edge_pairs_num:
            edge_pairs_num.extend([(edge[1], edge[0]) for edge in edge_pairs_num])

        processed_node_features = []
        for feats in node_features:
            flattened = []
            for idx, feat in enumerate(feats):
                if idx == 0 and isinstance(feat, np.ndarray):
                    flattened.extend(feat.flatten().tolist())
                elif isinstance(feat, np.ndarray):
                    flattened.extend(feat.flatten().tolist()) if feat.size > 1 else flattened.append(float(feat.item()))
                else:
                    flattened.append(float(feat))
            processed_node_features.append(flattened)

        node_feature_tensor = torch.tensor(processed_node_features, dtype=torch.float32)
        processed_edge_features = [[int(f) for f in feats] for feats in edge_features]

        if processed_edge_features:
             edge_feature_tensor = torch.tensor(processed_edge_features, dtype=torch.float32)
        else:
             edge_feature_tensor = torch.empty(0, 7)

        g.ndata['h'] = node_feature_tensor
        g.ndata['atom_type_and_id'] = torch.arange(len(node_ids_labels), dtype=torch.int64)

        if edge_feature_tensor.numel() > 0:
            g.edata['features'] = edge_feature_tensor

        if g.num_edges() > 0:
             g = dgl.add_reverse_edges(g, copy_ndata=True, copy_edata=True)

        if g.num_nodes() > 0:
            k = 9
            laplacian_pos = dgl.lap_pe(g, k=k, padding=True)
            laplacian_positional_embed_9_dim.append(laplacian_pos)
        else:
            laplacian_positional_embed_9_dim.append(torch.empty(0, 9))

        node_features_tensor_extract = g.ndata['h']

        if 'features' in g.edata:
             edge_features_tensor_extract = g.edata['features']
        else:
             edge_features_tensor_extract = torch.empty(0, 7)

        node_features_tensor_list.append(node_features_tensor_extract)
        edge_features_tensor_list.append(edge_features_tensor_extract)
        edge_pairs_list.append(edge_pairs_num)

    return node_features_tensor_list, edge_features_tensor_list, edge_pairs_list, laplacian_positional_embed_9_dim
