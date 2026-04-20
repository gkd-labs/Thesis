import os
import sys
import csv
import time
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from ogb.lsc import PCQM4Mv2Evaluator
from tqdm import tqdm

from model.chemgraphormer import ChemGraphormerClassifier   # ensure model folder is in your current directory

def move_batch_to_device(batch, device):
    """
    Move all relevant attributes of a PyG batch to the specified device.

    Args:
        batch: PyG batch object containing x, lap_pos, edge_index, edge_attr, batch, y
        device: Target device ("cuda" or "cpu")

    Returns:
        batch: PyG batch with all tensor attributes moved to device
    """
    for attr in ['x', 'lap_pos', 'edge_index', 'edge_attr', 'batch', 'y']:
        if hasattr(batch, attr):
            tensor = getattr(batch, attr)
            if isinstance(tensor, torch.Tensor):
                setattr(batch, attr, tensor.to(device))
    return batch


def build_optimizer(model, lr_multiplier):
    """
    Build AdamW optimizer with different learning rates for model components.

    Args:
        model: ChemGraphormerClassifier instance
        lr_multiplier: Multiplier for the base learning rates

    Returns:
        optimizer: torch.optim.AdamW instance
    """
    return torch.optim.AdamW([
        {"params": model.encoder.embed.parameters(), "lr": lr_multiplier * 1e-4},
        {"params": model.encoder.layers[0].parameters(), "lr": lr_multiplier * 9e-5},
        {"params": model.encoder.layers[1].parameters(), "lr": lr_multiplier * 8e-5},
        {"params": model.encoder.layers[2].parameters(), "lr": lr_multiplier * 7e-5},
        {"params": model.encoder.layers[3].parameters(), "lr": lr_multiplier * 6e-5},
        {"params": model.encoder.layers[4].parameters(), "lr": lr_multiplier * 5e-5},
        {"params": model.encoder.layers[5].parameters(), "lr": lr_multiplier * 4e-5},
        {"params": model.encoder.layers[6].parameters(), "lr": lr_multiplier * 3e-5},
        {"params": model.encoder.layers[7].parameters(), "lr": lr_multiplier * 2e-5},
        {"params": model.pool.parameters(), "lr": lr_multiplier * 1e-4},
        {"params": model.head.parameters(), "lr": lr_multiplier * 1e-3},
    ], weight_decay=5e-4)


def build_scheduler(optimizer, patience=10, factor=0.80, min_lr=1e-6):
    """
    Build ReduceLROnPlateau scheduler to decay LR based on validation performance.

    Args:
        optimizer: Optimizer to schedule
        patience: Number of epochs with no improvement before LR decay
        factor: Multiplicative factor of LR decay
        min_lr: Minimum allowed LR

    Returns:
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau instance
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr
    )


def count_parameters(model):
    """
    Print total and trainable parameters of a model.

    Args:
        model: torch.nn.Module
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total:,} total | {trainable:,} trainable")


def train_one_epoch(model, loader, optimizer, device, scaler):
    """
    Perform one epoch of training.

    Args:
        model: ChemGraphormerClassifier instance
        loader: DataLoader for training data
        optimizer: Optimizer
        device: Training device
        scaler: AMP GradScaler for mixed precision

    Returns:
        avg_loss: Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Train", leave=False, dynamic_ncols=True):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda"):
            pred = model(batch)
            loss = F.l1_loss(pred.squeeze(), batch.y.squeeze())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * batch.num_graphs
        del batch, pred, loss
        torch.cuda.empty_cache()
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate_one_epoch(model, loader, device, evaluator):
    """
    Validate model for one epoch.

    Args:
        model: ChemGraphormerClassifier instance
        loader: DataLoader for validation data
        device: Validation device
        evaluator: PCQM4Mv2Evaluator instance

    Returns:
        val_mae: Mean absolute error on validation set
    """
    model.eval()
    preds, labels = [], []
    for batch in tqdm(loader, desc="Val", leave=False, dynamic_ncols=True):
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(device_type="cuda"):
            pred = model(batch)
        preds.append(pred.cpu())
        labels.append(batch.y.cpu())
        del batch, pred
        torch.cuda.empty_cache()
    preds = torch.cat(preds, dim=0).view(-1)
    labels = torch.cat(labels, dim=0).view(-1)
    return evaluator.eval({"y_pred": preds, "y_true": labels})["mae"]


def get_resume_checkpoint(ckpt_dir):
    """
    Return path to the last checkpoint if it exists.

    Args:
        ckpt_dir: Directory containing checkpoints

    Returns:
        path: Full path to last_checkpoint.pt or None
    """
    path = os.path.join(ckpt_dir, "last_checkpoint.pt")
    return path if os.path.exists(path) else None


def run_training(train_path, valid_path, ckpt_dir, epoch_log_path, epochs=100, device="cuda"):
    """
    Full model training loop for ChemGraphormer with dynamic LR, batch size scheduling,
    mixed precision, checkpointing, and CSV logging.

    Args:
        train_path: Path to preprocessed training dataset (.pt)
        valid_path: Path to preprocessed validation dataset (.pt)
        ckpt_dir: Directory to save checkpoints
        epoch_log_path: CSV file path for logging epoch metrics
        epochs: Total number of epochs to train
        device: Training device ("cuda" or "cpu")
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    print("Loading datasets...")
    train_dataset = torch.load(train_path, map_location="cpu", weights_only=False)  # custom PyG dataset objects require full unpickling
    valid_dataset = torch.load(valid_path, map_location="cpu", weights_only=False)  # custom PyG dataset objects require full unpickling

    sample_graph = train_dataset[0]
    model = ChemGraphormerClassifier(
        d_node=sample_graph.x.size(1),
        d_edge=sample_graph.edge_attr.size(1),
        lap_dim=sample_graph.lap_pos.size(1),
        num_classes=1
    ).to(device)
    count_parameters(model)
    evaluator = PCQM4Mv2Evaluator()

    latest_ckpt = get_resume_checkpoint(ckpt_dir)
    start_epoch, best_checkpoints = 1, []
    optimizer, scheduler = None, None
    scaler = torch.amp.GradScaler('cuda')
    current_batch_size, current_lr_multiplier = 0, 0
    
    loaded_ckpt = None
    if latest_ckpt:
        loaded_ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(loaded_ckpt["model_state"])
        start_epoch = loaded_ckpt["epoch"] + 1
        best_checkpoints = loaded_ckpt.get("best_checkpoints", [])
        scaler.load_state_dict(loaded_ckpt.get("scaler_state", scaler.state_dict()))

    
    log_dir = os.path.dirname(epoch_log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    if start_epoch == 1:
        with open(epoch_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Val MAE', 'Epoch Time (min)',
                             'LR Embed', 'LR Layer0', 'LR Layer1', 'LR Layer2', 'LR Layer3',
                             'LR Layer4', 'LR Layer5', 'LR Layer6', 'LR Layer7', 'LR Pool', 'LR Head'])

    for epoch in range(start_epoch, epochs + 1):
        if 1 <= epoch <= 30: new_batch_size, lr_multiplier = 2048, 4.0
        elif 31 <= epoch <= 65: new_batch_size, lr_multiplier = 512, 1.0
        elif 66 <= epoch <= 92: new_batch_size, lr_multiplier = 256, 0.5
        else: new_batch_size, lr_multiplier = 128, 0.25

        if new_batch_size != current_batch_size or lr_multiplier != current_lr_multiplier or epoch == start_epoch:
            current_batch_size, current_lr_multiplier = new_batch_size, lr_multiplier
            train_loader = DataLoader(train_dataset, batch_size=new_batch_size, shuffle=True, pin_memory=True)
            valid_loader = DataLoader(valid_dataset, batch_size=new_batch_size, shuffle=False, pin_memory=True)
            optimizer = build_optimizer(model, lr_multiplier)
            scheduler = build_scheduler(optimizer)
            
            if loaded_ckpt and 'optimizer_state' in loaded_ckpt:
                saved_cfg = loaded_ckpt.get("train_cfg", {})
                if saved_cfg.get("lr_multiplier") == current_lr_multiplier and saved_cfg.get("batch_size") == current_batch_size:
                    optimizer.load_state_dict(loaded_ckpt["optimizer_state"])
                    scheduler.load_state_dict(loaded_ckpt["scheduler_state"])
                loaded_ckpt = None

        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_mae = validate_one_epoch(model, valid_loader, device, evaluator)
        scheduler.step(val_mae)
        epoch_minutes = (time.time() - start_time) / 60

        lrs = [g['lr'] for g in optimizer.param_groups]
        print(f"Epoch {epoch}: Train Loss={train_loss:.5f}, Val MAE={val_mae:.5f}, Time={epoch_minutes:.2f} min")
        with open(epoch_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.5f}", f"{val_mae:.5f}", f"{epoch_minutes:.2f}"] +
                            [f"{lr:.2e}" for lr in lrs])

        if len(best_checkpoints) < 2 or val_mae < max(b[0] for b in best_checkpoints):
            ckpt_path = os.path.join(ckpt_dir, f"chemgraphormer_best_model_{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
            best_checkpoints.append((val_mae, ckpt_path))
            best_checkpoints.sort(key=lambda x: x[0])
            if len(best_checkpoints) > 2:
                _, worst_path = best_checkpoints.pop()
                if os.path.exists(worst_path):
                    os.remove(worst_path)

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_checkpoints": best_checkpoints,
            "train_cfg": {"batch_size": current_batch_size, "lr_multiplier": current_lr_multiplier},
        }, os.path.join(ckpt_dir, "last_checkpoint.pt"))

    print("Training complete. Top checkpoints:")
    for val, path in best_checkpoints:
        print(f"{path} | Val MAE: {val:.5f}")