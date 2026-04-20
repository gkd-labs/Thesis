import sys
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from model.chemgraphormer_ablate_gate_init_one import ChemGraphormerClassifier
from tqdm import tqdm

def load_model_from_checkpoint(checkpoint_path, model_params, device):
    model = ChemGraphormerClassifier(**model_params).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def predict_with_best_model(dataset_path, best_checkpoint_path, device="cuda"):
    print("Loading dataset...")
    dataset = torch.load(dataset_path, map_location="cpu")

    data_sample = dataset[0]
    model_params = {
        'd_node': data_sample.x.size(1),
        'd_edge': data_sample.edge_attr.size(1),
        'lap_dim': data_sample.lap_pos.size(1),
        'num_classes': 1,
        'd_model': 256,
        'n_heads': 8,
        'num_layers': 6,
        'd_ff': 1024,
        'num_freqs': 16,
        'dropout': 0.2
    }

    # Load the best model
    model = load_model_from_checkpoint(best_checkpoint_path, model_params, device)

    data_loader = DataLoader(dataset, batch_size=128, shuffle=False)
    all_predictions, all_targets = [], []

    # Run inference
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Best Model Prediction", leave=False, dynamic_ncols=True):
            for attr in ['x', 'lap_pos', 'edge_index', 'edge_attr', 'batch']:
                if hasattr(batch, attr):
                    setattr(batch, attr, getattr(batch, attr).to(device))

            preds = model(batch)
            all_predictions.append(preds.cpu().view(-1))

            # Collect ground truth targets if present
            if hasattr(batch, "y"):
                all_targets.append(batch.y.view(-1).cpu())

    predictions = torch.cat(all_predictions, dim=0)

    # Compute MAE only if targets exist
    if len(all_targets) > 0:
        targets = torch.cat(all_targets, dim=0)
        mae = F.l1_loss(predictions, targets).item()
        print(f"\nOverall Mean Absolute Error (MAE): {mae:.6f}")
        return predictions, mae
    else:
        print("\nNo targets found in dataset. Returning predictions only.")
        return predictions, None
