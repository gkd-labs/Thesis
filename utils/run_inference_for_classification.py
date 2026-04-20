"""
Inference Script for Fine-Tuned ChemGraphormer - Classification Mode
Loads the best fine-tuned model and runs predictions on new/test data.
Supports computing classification metrics (Accuracy, Precision, Recall, F1, Specificity, AUC, Kappa, MCC) if labels are provided.
Assumes binary classification (num_classes=2); extend for multi-class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import os
import csv
import sys
import numpy as np
from sklearn.metrics import roc_auc_score, cohen_kappa_score, matthews_corrcoef

from model.chemgraphormer import ChemGraphormerClassifier   # make sure model folder is present in the current working directory


def move_batch_to_device(batch, device):
    """
    Move batch attributes to specified device.
    """
    for attr in ['x', 'lap_pos', 'edge_index', 'edge_attr', 'batch', 'y']:
        if hasattr(batch, attr):
            tensor = getattr(batch, attr)
            if isinstance(tensor, torch.Tensor):
                setattr(batch, attr, tensor.to(device))
    return batch


@torch.no_grad()
def predict(model, dataset, device, batch_size=64, return_probs=False):
    """
    Run inference on a dataset and return metrics and predictions.
    Assumes binary classification; returns class predictions (0/1) or probs for positive class.
    """
    model.eval()
    metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'auc', 'kappa', 'mcc']
    all_labels = []

    has_labels = len(dataset) > 0 and hasattr(dataset[0], 'y')

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    run_probs, run_preds, run_labels = [], [], []

    for batch in tqdm(loader, desc="Prediction", leave=False):
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast("cuda"):
            logits = model(batch)  # Logits [batch_size, num_classes]
        probs_class1 = F.softmax(logits, dim=1)[:, 1].cpu()  # probability for class 1
        class_preds = torch.argmax(logits, dim=1).cpu()
        run_probs.append(probs_class1)
        run_preds.append(class_preds)
        if has_labels:
            run_labels.append(batch.y.long().cpu())
        del batch, logits
        torch.cuda.empty_cache()

    full_probs = torch.cat(run_probs).numpy()  # probabilities for class 1
    full_preds = torch.cat(run_preds).numpy()  # hard labels (0 or 1)

    if has_labels:
        full_labels = torch.cat(run_labels).numpy()
        all_labels.append(full_labels)

        accuracy = (full_preds == full_labels).mean()

        tp = np.sum((full_preds == 1) & (full_labels == 1))
        fp = np.sum((full_preds == 1) & (full_labels == 0))
        fn = np.sum((full_preds == 0) & (full_labels == 1))
        tn = np.sum((full_preds == 0) & (full_labels == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        auc = roc_auc_score(full_labels, full_probs) if len(np.unique(full_labels)) > 1 else 0

        kappa = cohen_kappa_score(full_labels, full_preds)
        mcc = matthews_corrcoef(full_labels, full_preds) if len(np.unique(full_labels)) > 1 else 0

        run_metrics = {
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "specificity": specificity, "auc": auc, "kappa": kappa, "mcc": mcc
        }

        return {"preds": torch.tensor(full_preds), "probs": torch.tensor(full_probs), "metrics": run_metrics}

    return torch.tensor(full_preds), torch.tensor(full_probs)


def run_inference(model_path, test_path, output_csv_path, device="cuda", batch_size=64, num_classes=2, return_probs=False):
    """
    Load already trained classification model and run predictions on test data.
    Saves predictions (classes or probs) and metrics to CSV if labels present.
    """
    print("Loading test dataset...")
    test_dataset = torch.load(test_path, weights_only=False, map_location="cpu")
    print(f"Test dataset size: {len(test_dataset)}")

    sample_graph = test_dataset[0]
    d_node = sample_graph.x.size(1)
    d_edge = sample_graph.edge_attr.size(1)
    lap_dim = sample_graph.lap_pos.size(1)

    model = ChemGraphormerClassifier(
        d_node=d_node, d_edge=d_edge, lap_dim=lap_dim, num_classes=num_classes,
        d_model=512, n_heads=16, num_layers=8, d_ff=1536, num_freqs=16, dropout=0.2
    ).to(device)

    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, weights_only=False, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded model from: {model_path}")
    else:
        raise FileNotFoundError("Model checkpoint not found. Check path.")

    has_labels = len(test_dataset) > 0 and hasattr(test_dataset[0], 'y')
    if has_labels:
        results = predict(model, test_dataset, device, batch_size=batch_size, return_probs=return_probs)
        preds = results["preds"]
        probs_class1 = results["probs"]
        metrics = results["metrics"]

        print("Test Metrics:")
        for key, val in metrics.items():
            print(f"  {key.capitalize()}: {val:.5f}")

        # Save to CSV with hard label + probability for both classes
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        with open(output_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for key, val in metrics.items():
                writer.writerow([key.capitalize(), f"{val:.5f}"])
            writer.writerow([])
            writer.writerow(['Predicted Label', 'Predicted Inactive(0) Probability', 'Predicted Active(1) Probability'])
            for label, prob1 in zip(preds, probs_class1):
                prob0 = 1.0 - prob1
                writer.writerow([label.item(), f"{prob0:.5f}", f"{prob1:.5f}"])
    else:
        preds, probs_class1 = predict(model, test_dataset, device, batch_size=batch_size, return_probs=return_probs)
        print("Predictions completed (no labels for metrics).")

        # Save predictions with probabilities
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        with open(output_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Predicted Label', 'Probability Class 0', 'Probability Class 1'])
            for label, prob1 in zip(preds, probs_class1):
                prob0 = 1.0 - prob1
                writer.writerow([label.item(), f"{prob0:.5f}", f"{prob1:.5f}"])

    print(f"Predictions and metrics saved to: {output_csv_path}")
    return preds, probs_class1