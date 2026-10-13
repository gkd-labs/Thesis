"""
Fine-Tuning Script for ChemGraphormer - Classification Mode.
Trains on full training set and evaluates on full validation set (internal validation).
Metrics: Accuracy, Precision, Recall, F1, Specificity, AUC, Cohen's Kappa, MCC.
Uses MCC for early stopping and model selection.
Unfreezes only head + pool for adaptation.
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

from model.chemgraphormer import ChemGraphormerClassifier   # ensure model folder is present in your current working directory

def move_batch_to_device(batch, device):
    """Move batch attributes to specified device."""
    for attr in ['x', 'lap_pos', 'edge_index', 'edge_attr', 'batch', 'y']:
        if hasattr(batch, attr):
            tensor = getattr(batch, attr)
            if isinstance(tensor, torch.Tensor):
                setattr(batch, attr, tensor.to(device))
    return batch

def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def train_one_epoch(model, loader, optimizer, device, scaler):
    """Train the model (CrossEntropy loss for classification)."""
    model.train()
    total_loss = 0

    # Class weights from training label frequencies (inverse frequency: w_c = N / (2 * n_c))
    train_labels = torch.tensor([int(d.y.view(-1)[0].item()) for d in loader.dataset])
    class_counts = torch.bincount(train_labels, minlength=2).float()
    class_weights = (class_counts.sum() / (2.0 * class_counts.clamp(min=1))).to(device)

    for batch in tqdm(loader, desc="Train", leave=False, dynamic_ncols=True):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda"):
            pred = model(batch)
            loss = F.cross_entropy(pred, batch.y.long().view(-1), weight=class_weights)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * batch.num_graphs
        del batch, pred, loss
        torch.cuda.empty_cache()

    return total_loss / len(loader.dataset)

@torch.no_grad()
def validate_one_epoch(model, loader, device):
    """Validate the model and return classification metrics."""
    model.eval()
    preds, probs, labels = [], [], []

    for batch in tqdm(loader, desc="Val", leave=False, dynamic_ncols=True):
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast("cuda"):
            pred = model(batch)

        probs.append(F.softmax(pred, dim=1)[:, 1].cpu())
        preds.append(torch.argmax(pred, dim=1).cpu())
        labels.append(batch.y.long().cpu())

        del batch, pred
        torch.cuda.empty_cache()

    probs = torch.cat(probs).numpy()
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()

    accuracy = (preds == labels).mean()
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0
    kappa = cohen_kappa_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds) if len(np.unique(labels)) > 1 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "auc": auc,
        "kappa": kappa,
        "mcc": mcc
    }

def build_finetune_optimizer(model, lr=1e-4, weight_decay=5e-3):
    """Build AdamW optimizer for fine-tuning."""
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

# Run AChEi Training

def run_finetuning(
    train_path,
    valid_path,
    encoder_ckpt_path,
    finetune_ckpt_dir,
    finetune_log_csv_path,
    batch_size=64,
    epochs=50,
    device="cuda",
    num_classes=2,
    n_heads=16,
    num_layers=8,
    lr=1e-4,
    dropout=0.3,
    patience=5,
    min_delta=1e-4
):
    """Run classification fine-tuning on training set and evaluate on validation set."""
    os.makedirs(finetune_ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(finetune_log_csv_path), exist_ok=True)

    print("Loading datasets...")
    train_dataset = torch.load(train_path, weights_only=False, map_location="cpu")
    val_dataset = torch.load(valid_path, weights_only=False, map_location="cpu")
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    sample_graph = train_dataset[0]
    d_node = sample_graph.x.size(1)
    d_edge = sample_graph.edge_attr.size(1)
    lap_dim = sample_graph.lap_pos.size(1)

    model = ChemGraphormerClassifier(
        d_node=d_node,
        d_edge=d_edge,
        lap_dim=lap_dim,
        num_classes=num_classes,
        d_model=512,
        n_heads=n_heads,
        num_layers=num_layers,
        d_ff=1536,
        num_freqs=16,
        dropout=dropout
    ).to(device)

    # Load pretrained encoder
    if encoder_ckpt_path and os.path.exists(encoder_ckpt_path):
        encoder_state = torch.load(encoder_ckpt_path, map_location=device, weights_only=True)
        model.encoder.load_state_dict(encoder_state)
        print("ChemGraphormer encoder state loaded successfully")
    else:
        raise FileNotFoundError(f"Encoder checkpoint not found at: {encoder_ckpt_path}")

    # Reinitialize head
    for m in model.head.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    print("Head reinitialized.")

    # Reinitialize the pool
    for m in model.pool.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    print("Pool reinitialized.")

    # Freeze encoder + embeddings and only unfreeze head + pool for adaptation
    for name, param in model.named_parameters():
        if ('head' in name) or ('pool' in name):
            param.requires_grad = True
        else:
            param.requires_grad = False

    print("Trainable parameters:")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"{n} | shape: {tuple(p.shape)} | params: {p.numel()}")

    # Use count_parameters to get total and trainable
    total, trainable = count_parameters(model)
    print(f"Model Parameters Size: {total:,} total | {trainable:,} trainable")

    optimizer = build_finetune_optimizer(model, lr=lr)
    scaler = torch.amp.GradScaler("cuda")

    header = [
        'Epoch', 'Train Loss', 'Val Accuracy', 'Val Precision', 'Val Recall',
        'Val F1', 'Val Specificity', 'Val AUC', 'Val Kappa', 'Val MCC'
    ]
    with open(finetune_log_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    print("AChEi Predictive Model Training Begins...")
    best_val_mcc = -1.0
    patience_counter = 0
    best_path = os.path.join(finetune_ckpt_dir, "chemgraphormer_finetuned_best.pt")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_metrics = validate_one_epoch(model, val_loader, device)
        val_mcc = val_metrics["mcc"]

        print(
            f"Epoch {epoch}: Train Loss={train_loss:.5f}, "
            f"Val MCC={val_mcc:.5f}, Val AUC={val_metrics['auc']:.5f}, "
            f"Val F1={val_metrics['f1']:.5f}"
        )

        # Log metrics
        row = [epoch, f"{train_loss:.5f}"]
        for key in ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'auc', 'kappa', 'mcc']:
            row.append(f"{val_metrics[key]:.5f}")
        with open(finetune_log_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # Early stopping & best model save
        if val_mcc > best_val_mcc + min_delta:
            best_val_mcc = val_mcc
            patience_counter = 0
            if os.path.exists(best_path):
                os.remove(best_path)
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. No Val MCC improvement for {patience} epochs")
                break

    # Load best model for final evaluation
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    final_val_metrics = validate_one_epoch(model, val_loader, device)

    print("Final Val Metrics:")
    for key, val in final_val_metrics.items():
        print(f"  {key.capitalize()}: {val:.5f}")

    print(f"Fine-tuning complete. Best model saved to {best_path} with Val MCC={best_val_mcc:.5f}")