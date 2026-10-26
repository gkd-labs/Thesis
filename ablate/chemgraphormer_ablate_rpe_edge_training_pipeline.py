import os
import csv
import time
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader
from model.chemgraphormer_ablate_rpe_edge import ChemGraphormerClassifier
from tqdm import tqdm

def move_batch_to_device(batch, device):
    for attr in ['x', 'lap_pos', 'edge_index', 'edge_attr', 'batch', 'y']:
        if hasattr(batch, attr):
            tensor = getattr(batch, attr)
            if isinstance(tensor, torch.Tensor):
                setattr(batch, attr, tensor.to(device))
    return batch

def random_sign_flip(lap_pos, batch_idx, num_graphs):
    """Flip each Laplacian eigenvector independently, per graph (training-time augmentation)."""
    signs = torch.randint(0, 2, (num_graphs, lap_pos.size(1)),
                          device=lap_pos.device).to(lap_pos.dtype) * 2 - 1
    return lap_pos * signs[batch_idx]

def build_optimizer(model, lr_multiplier):
    return torch.optim.AdamW([
        {"params": model.encoder.embed.parameters(), "lr": lr_multiplier * 1e-4},
        {"params": model.encoder.layers[0].parameters(), "lr": lr_multiplier * 7e-5},
        {"params": model.encoder.layers[1].parameters(), "lr": lr_multiplier * 6e-5},
        {"params": model.encoder.layers[2].parameters(), "lr": lr_multiplier * 5e-5},
        {"params": model.encoder.layers[3].parameters(), "lr": lr_multiplier * 4e-5},
        {"params": model.encoder.layers[4].parameters(), "lr": lr_multiplier * 3e-5},
        {"params": model.encoder.layers[5].parameters(), "lr": lr_multiplier * 2e-5},
        {"params": model.head.parameters(), "lr": lr_multiplier * 1e-3},
    ], weight_decay=5e-4)

def build_scheduler(optimizer, patience=5, factor=0.7, min_lr=8e-7):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total:,} total | {trainable:,} trainable")

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    scaler = torch.amp.GradScaler('cuda')
    for batch in tqdm(loader, desc="Train", leave=False, dynamic_ncols=True):
        batch = move_batch_to_device(batch, device)
        batch.lap_pos = random_sign_flip(batch.lap_pos, batch.batch, batch.num_graphs)
        optimizer.zero_grad()
        with autocast('cuda'):
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), batch.y.view(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * batch.num_graphs
        del batch, pred, loss
        torch.cuda.empty_cache()
    return total_loss / len(loader.dataset)

@torch.no_grad()
def validate_one_epoch(model, loader, device):
    model.eval()
    total_mae = 0
    num_samples = 0
    for batch in tqdm(loader, desc="Val", leave=False, dynamic_ncols=True):
        batch = move_batch_to_device(batch, device)
        with autocast('cuda'):
            pred = model(batch)
        mae = F.l1_loss(pred.view(-1), batch.y.view(-1), reduction='sum').item()
        total_mae += mae
        num_samples += batch.y.numel()
        del batch, pred
        torch.cuda.empty_cache()
    return total_mae / num_samples

def get_resume_checkpoint(ckpt_dir):
    ckpt_path = os.path.join(ckpt_dir, "last_checkpoint.pt")
    return ckpt_path if os.path.exists(ckpt_path) else None

def run_rpe_edge_ablation(train_path, valid_path, ckpt_dir, epoch_log_path, epochs=100, device="cuda", early_stopping_patience=15):
    os.makedirs(ckpt_dir, exist_ok=True)
    print("Loading datasets...")
    train_dataset = torch.load(train_path, map_location="cpu")
    valid_dataset = torch.load(valid_path, map_location="cpu")

    sample_graph = train_dataset[0]
    model = ChemGraphormerClassifier(
        d_node=sample_graph.x.size(1),
        d_edge=sample_graph.edge_attr.size(1),
        lap_dim=sample_graph.lap_pos.size(1),
        num_classes=1
    ).to(device)

    count_parameters(model)

    latest_ckpt = get_resume_checkpoint(ckpt_dir)
    start_epoch = 1
    best_val_mae = float("inf")
    best_path = None

    optimizer = None
    scheduler = None
    scaler = torch.amp.GradScaler('cuda')
    current_batch_size = 0
    current_lr_multiplier = 0
    no_improve_epochs = 0  

    if latest_ckpt:
        ckpt = torch.load(latest_ckpt, map_location=device)
        print(f"Resuming from checkpoint: {latest_ckpt} (epoch {ckpt['epoch']})")
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_mae = ckpt.get("best_val_mae", best_val_mae)
        best_path = ckpt.get("best_path", None)
        scaler.load_state_dict(ckpt.get("scaler_state", scaler.state_dict()))

    os.makedirs(os.path.dirname(epoch_log_path), exist_ok=True)
    if start_epoch == 1:
        with open(epoch_log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Epoch', 'Train Loss', 'Val MAE', 'Epoch Time (min)',
                'LR Embed', 'LR Layer 0', 'LR Layer 1', 'LR Layer 2',
                'LR Layer 3', 'LR Layer 4', 'LR Layer 5', 'LR Head'
            ])

    for epoch in range(start_epoch, epochs + 1):
        # Determine current batch size and LR multiplier
        if 1 <= epoch <= 80:
            new_batch_size = 2048
            lr_multiplier = 4.0
        else:
            new_batch_size = 128
            lr_multiplier = 0.25

        # Rebuild loaders/optimizers 
        if new_batch_size != current_batch_size or lr_multiplier != current_lr_multiplier or epoch == start_epoch:
            print(f"Epoch {epoch}: Switching to batch size {new_batch_size} and LR multiplier {lr_multiplier}.")
            current_batch_size = new_batch_size
            current_lr_multiplier = lr_multiplier

            train_loader = DataLoader(train_dataset, batch_size=new_batch_size, shuffle=True, pin_memory=True)
            valid_loader = DataLoader(valid_dataset, batch_size=new_batch_size, shuffle=False, pin_memory=True)
            optimizer = build_optimizer(model, lr_multiplier)
            scheduler = build_scheduler(optimizer, patience=5, factor=0.7, min_lr=8e-7)

            if latest_ckpt and 'optimizer_state' in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
                scheduler.load_state_dict(ckpt["scheduler_state"])
                latest_ckpt = None

        # ===== Training and validation =====
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_mae = validate_one_epoch(model, valid_loader, device)
        scheduler.step(val_mae)
        epoch_minutes = (time.time() - start_time) / 60.0

        print(f"Epoch {epoch}: Train Loss={train_loss:.5f}, Val MAE={val_mae:.5f}, Time={epoch_minutes:.2f} min")

        # ===== Fixed LR Logging =====
        lr_embed = optimizer.param_groups[0]['lr']
        lr_layer0 = optimizer.param_groups[1]['lr']
        lr_layer1 = optimizer.param_groups[2]['lr']
        lr_layer2 = optimizer.param_groups[3]['lr']
        lr_layer3 = optimizer.param_groups[4]['lr']
        lr_layer4 = optimizer.param_groups[5]['lr']
        lr_layer5 = optimizer.param_groups[6]['lr']
        lr_head  = optimizer.param_groups[7]['lr']

        with open(epoch_log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.5f}",
                f"{val_mae:.5f}",
                f"{epoch_minutes:.2f}",
                f"{lr_embed:.2e}",
                f"{lr_layer0:.2e}",
                f"{lr_layer1:.2e}",
                f"{lr_layer2:.2e}",
                f"{lr_layer3:.2e}",
                f"{lr_layer4:.2e}",
                f"{lr_layer5:.2e}",
                f"{lr_head:.2e}"
            ])

        # Save last checkpoint
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_mae": best_val_mae,
            "best_path": best_path,
        }, os.path.join(ckpt_dir, "last_checkpoint.pt"))

        # ===== Early stopping logic =====
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            no_improve_epochs = 0  # reset patience counter
            if best_path:
                os.remove(best_path)
            best_path = os.path.join(ckpt_dir, "chemmy_best_model.pt")
            torch.save(model.state_dict(), best_path)
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}. Best Val MAE={best_val_mae:.5f}")
                break

    print(f"Training complete. Best model saved to {best_path} with Val MAE={best_val_mae:.5f}")
