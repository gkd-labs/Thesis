import sys
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from ogb.lsc import PCQM4Mv2Evaluator

from model.chemgraphormer import ChemGraphormerClassifier    # make sure model folder is present in the current working directory
from tqdm import tqdm


def load_model_from_checkpoint(checkpoint_path, model_params, device):
    """Loads a model from a saved checkpoint."""
    model = ChemGraphormerClassifier(**model_params).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

# FOR SIMPEL WEIGHTS AVERAGE ENSEMBLE
def run_simple_ensemble(dataset_path, checkpoint1_path, checkpoint2_path, device="cuda"):
    """Runs a simple averaging ensemble on two models."""
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
        'dropout': 0.0
    }

    # Load model checkpoints at epochs 92 and 95. At these epochs, train loss were 0.11597 and 0.11474 respectively. Var mae was also 0.10663 and 0.10626 respectively
    model1 = load_model_from_checkpoint(checkpoint1_path, model_params, device)
    model2 = load_model_from_checkpoint(checkpoint2_path, model_params, device)

    valid_loader = DataLoader(dataset, batch_size=128, shuffle=False)
    all_ensemble_preds = []
    evaluator = PCQM4Mv2Evaluator()

    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Simple Ensemble For Val Prediction", leave=False, dynamic_ncols=True):
            batch.to(device)
            pred1 = model1(batch)
            pred2 = model2(batch)
            ensemble_pred = (pred1 + pred2) / 2
            all_ensemble_preds.append(ensemble_pred.cpu())

    all_ensemble_preds = torch.cat(all_ensemble_preds, dim=0).view(-1)
    true_labels = torch.cat([batch.y.cpu() for batch in valid_loader], dim=0).view(-1)

    mae = evaluator.eval({"y_pred": all_ensemble_preds, "y_true": true_labels})["mae"]
    print(f"Ensemble MAE using Simple Averaging Method: {mae:.5f}")


# FOR WEIGHTED VAL MAE BASED WEIGHTS ENSEMBLE
def run_weighted_ensemble(dataset_path, checkpoint1_path, checkpoint2_path, mae1, mae2, device="cuda"):
    """Runs a weighted averaging ensemble on two models."""
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
        'dropout': 0.0
    }

    # Load models
    model1 = load_model_from_checkpoint(checkpoint1_path, model_params, device)
    model2 = load_model_from_checkpoint(checkpoint2_path, model_params, device)

    # Calculate weights based on the inverse of MAEs
    weight1 = 1 / mae1
    weight2 = 1 / mae2
    total_weight = weight1 + weight2
    normalized_weight1 = weight1 / total_weight
    normalized_weight2 = weight2 / total_weight

    valid_loader = DataLoader(dataset, batch_size=128, shuffle=False)
    all_ensemble_preds = []
    evaluator = PCQM4Mv2Evaluator()

    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Weighted Ensemble Val Prediction", leave=False, dynamic_ncols=True):
            batch.to(device)
            pred1 = model1(batch)
            pred2 = model2(batch)
            ensemble_pred = (pred1 * normalized_weight1) + (pred2 * normalized_weight2)
            all_ensemble_preds.append(ensemble_pred.cpu())

    all_ensemble_preds = torch.cat(all_ensemble_preds, dim=0).view(-1)
    true_labels = torch.cat([batch.y.cpu() for batch in valid_loader], dim=0).view(-1)

    mae = evaluator.eval({"y_pred": all_ensemble_preds, "y_true": true_labels})["mae"]
    print(f"Ensemble MAE (Weighted Averaging): {mae:.5f}")
