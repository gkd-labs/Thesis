import sys
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from model.chemgraphormer import ChemGraphormerClassifier  # make sure model folder is present in the current working directory
from tqdm import tqdm

def load_model_from_checkpoint(checkpoint_path, model_params, device):
    model = ChemGraphormerClassifier(**model_params).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def predict_with_simple_ensemble(dataset_path, checkpoint1_path, checkpoint2_path, device="cuda"):
    dataset = torch.load(dataset_path, map_location="cpu")
    data_sample = dataset[0]
    model_params = {
        'd_node': data_sample.x.size(1),
        'd_edge': data_sample.edge_attr.size(1),
        'lap_dim': data_sample.lap_pos.size(1),
        'num_classes': 1,
        'd_model': 512,
        'n_heads': 16,
        'num_layers': 8,
        'd_ff': 1536,
        'num_freqs': 16,
        'dropout': 0.2
    }

    model1 = load_model_from_checkpoint(checkpoint1_path, model_params, device)
    model2 = load_model_from_checkpoint(checkpoint2_path, model_params, device)

    unlabeled_loader = DataLoader(dataset, batch_size=128, shuffle=False)
    all_ensemble_preds = []

    with torch.no_grad():
        for batch in tqdm(unlabeled_loader, desc="Simple Ensemble Prediction", leave=False, dynamic_ncols=True):
            for attr in ['x', 'lap_pos', 'edge_index', 'edge_attr', 'batch']:
                if hasattr(batch, attr):
                    setattr(batch, attr, getattr(batch, attr).to(device))

            pred1 = model1(batch)
            pred2 = model2(batch)
            ensemble_pred = (pred1 + pred2) / 2
            all_ensemble_preds.append(ensemble_pred.cpu())

    return torch.cat(all_ensemble_preds, dim=0).view(-1)

def predict_with_weighted_ensemble(dataset_path, checkpoint1_path, checkpoint2_path, device="cuda"):
    mae1 = 0.10663
    mae2 = 0.10626

    print("Loading datasets...")
    dataset = torch.load(dataset_path, map_location="cpu")

    data_graph = dataset[0]
    model_params = {
        'd_node': data_graph.x.size(1),
        'd_edge': data_graph.edge_attr.size(1),
        'lap_dim': data_graph.lap_pos.size(1),
        'num_classes': 1,
        'd_model': 512,
        'n_heads': 16,
        'num_layers': 8,
        'd_ff': 1536,
        'num_freqs': 16,
        'dropout': 0.2
    }

    model1 = load_model_from_checkpoint(checkpoint1_path, model_params, device)
    model2 = load_model_from_checkpoint(checkpoint2_path, model_params, device)

    weight1 = 1 / mae1
    weight2 = 1 / mae2
    total_weight = weight1 + weight2
    normalized_weight1 = weight1 / total_weight
    normalized_weight2 = weight2 / total_weight

    testing_loader = DataLoader(dataset, batch_size=128, shuffle=False)
    all_ensemble_preds = []

    with torch.no_grad():
        for batch in tqdm(testing_loader, desc="Weighted Ensemble Prediction", leave=False, dynamic_ncols=True):
            for attr in ['x', 'lap_pos', 'edge_index', 'edge_attr', 'batch']:
                if hasattr(batch, attr):
                    setattr(batch, attr, getattr(batch, attr).to(device))

            pred1 = model1(batch)
            pred2 = model2(batch)
            ensemble_pred = (pred1 * normalized_weight1) + (pred2 * normalized_weight2)
            all_ensemble_preds.append(ensemble_pred.cpu())

    return torch.cat(all_ensemble_preds, dim=0).view(-1)