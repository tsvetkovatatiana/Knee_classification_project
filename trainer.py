import time
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from args import get_args
from utils import (
    calculate_metrics,
    plot_loss_curve,
    save_checkpoint
)
from torch.cuda.amp import GradScaler
from contextlib import nullcontext


def train_model(model, train_loader, val_loader, cur_fold, device):
    """
    Train the model with mixed precision and backbone freezing.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        cur_fold: Current fold number
        device: torch.device passed from main
    """
    args = get_args()

    model = model.to(device)

    # Enable Automatic Mixed Precision (AMP) only when running on a CUDA-capable GPU.
    # AMP allows certain operations to run in float16 (half precision) instead of float32,
    # which speeds up training and reduces memory
    use_amp = device.type == "cuda"
    amp_context = (lambda: torch.amp.autocast("cuda")) if use_amp else nullcontext

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler() if use_amp else None

    # --- Scheduler setup ---
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(args.epochs), eta_min=args.min_lr)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=args.gamma, patience=3, min_lr=args.min_lr)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        scheduler = None

    # Initialize tracking variables
    best_val_metric = None
    best_model_path = None
    train_losses = []
    val_losses = []

    # Freeze backbone layers for first few epochs
    freeze_epochs = min(4, int(args.epochs) // 2)
    if hasattr(model, "backbone"):
        print(f"[INFO] Freezing backbone for first {freeze_epochs} epochs.")
        for param in model.backbone.parameters():
            param.requires_grad = False

    for epoch in range(int(args.epochs)):
        epoch_start = time.time()

        model.train()
        epoch_training_loss = 0.0

        # Unfreeze backbone after freeze_epochs
        if epoch == freeze_epochs and hasattr(model, "backbone"):
            print("[INFO] Unfreezing backbone layers for fine-tuning.")
            for param in model.backbone.parameters():
                param.requires_grad = True

        # TRAIN LOOP
        for batch in train_loader:
            inputs = batch["img"].float().to(device, non_blocking=True)
            targets = batch["label"].long().to(device, non_blocking=True)

            # optimizer to zero
            optimizer.zero_grad(set_to_none=True)

            # mixed precision
            with amp_context():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # accumulate training loss (scalar)
            epoch_training_loss += loss.item()

            # free batch-level tensors proactively
            del inputs, targets, outputs, loss

        # End of epoch metrics & logging
        avg_train_loss = epoch_training_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch + 1}/{args.epochs} | Train loss: {avg_train_loss:.4f}")
        print(f"Epoch {epoch + 1} took {epoch_time:.2f} seconds")

        # VALIDATION
        metrics, y_true, y_pred, avg_val_loss = validate_model(model, val_loader, criterion, device)
        ba = metrics['balanced_acc']
        val_losses.append(avg_val_loss)

        best_ba, best_model_path = save_checkpoint(
            cur_fold,
            epoch,
            model,
            y_true,
            y_pred,
            ba,
            best_val_metric=best_val_metric,
            prev_model_path=best_model_path,
            comparator="gt",
            save_dir=args.output_dir,
        )
        best_val_metric = best_ba

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(ba)
            else:
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            print(f"[LR] Current learning rate: {current_lr:.6f}")

        plot_loss_curve(train_losses, val_losses, save_dir=args.output_dir, fold=cur_fold)

        # Clean up and free memory
        gc.collect()
        torch.cuda.empty_cache()

    return best_model_path


def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []

    use_amp = device.type == "cuda"
    amp_context = (lambda: torch.amp.autocast("cuda")) if use_amp else nullcontext

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["img"].float().to(device, non_blocking=True)
            targets = batch["label"].long().squeeze().to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            with amp_context():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            val_loss += loss.item()
            predictions = nn.functional.softmax(outputs, dim=1)
            pred_targets = predictions.max(dim=1)[1]

            all_preds.append(pred_targets.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            del inputs, targets, outputs, loss

    if len(all_preds) == 0:
        return 0.0, np.array([]), np.array([])

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    metrics = calculate_metrics(all_targets, all_preds)
    avg_val_loss = val_loss / len(val_loader)

    print(f"Validation Loss: {val_loss / len(val_loader):.4f} | "
          f"Balanced Accuracy: {metrics['balanced_acc']:.4f} | "
          f"F1 (micro): {metrics['f1_micro']:.4f} | "
          f"F1 (macro): {metrics['f1_macro']:.4f} | "
          f"F1 (weighted): {metrics['f1_weighted']:.4f} | ")
    print("=" * 60)

    return metrics, all_targets, all_preds, avg_val_loss
