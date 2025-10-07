import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import numpy as np

from args import get_args


def train_model(model, train_loader, val_loader, fold):
    args = get_args()
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model = model.to(device)

    # defining the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Track the best model by accuracy
    best_bal_acc = 0.0
    best_model_path = os.path.join(args.output_dir, f"best_model_fold_{fold}.pth")

    # Tracking metrics
    train_losses, val_losses = [], []
    train_bal_accs, val_bal_accs = [], []
    train_roc_aucs, val_roc_aucs = [], []
    train_avg_precs, val_avg_precs = [], []

    # starting to iterate through epochs
    for epoch in range(args.epochs):
        # starting the training -> setting the model to training mode
        model.train()
        epoch_training_loss = 0

        all_train_preds, all_train_labels, all_train_probs = [], [], []

        for batch in train_loader:
            inputs = batch['img']
            targets = batch['label']

            if not isinstance(inputs, torch.Tensor):
                inputs = torch.tensor(inputs, dtype=torch.float32)
            if not isinstance(targets, torch.Tensor):
                targets = torch.tensor(targets, dtype=torch.long)

            # Ensure targets are 1D
            if targets.dim() > 1:
                targets = targets.squeeze()

            inputs, targets = inputs.to(device), targets.to(device)

            # resetting the gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_training_loss += loss.item()

            # Get probabilities and predictions
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_train_probs.append(probs.cpu().detach().numpy())
            all_train_preds.extend(preds.cpu().detach().numpy())
            all_train_labels.extend(targets.cpu().detach().numpy())

        avg_train_loss = epoch_training_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        print("Epoch: {}: {}".format((epoch + 1), epoch_training_loss / len(train_loader)))

        all_train_probs = np.vstack(all_train_probs)

        # Compute training metrics
        train_metrics = calculate_train_accuracy_metrics(
            all_train_labels, all_train_preds, all_train_probs
        )
        train_bal_accs.append(train_metrics["balanced_accuracy"])
        train_roc_aucs.append(train_metrics["roc_auc"])
        train_avg_precs.append(train_metrics["avg_precision"])

        # Evaluation phase
        val_loss = validation_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        val_metrics = calculate_val_accuracy_metrics(model, val_loader, device)
        val_bal_accs.append(val_metrics['balanced_accuracy'])
        val_roc_aucs.append(val_metrics['roc_auc'])
        val_avg_precs.append(val_metrics['avg_precision'])

        print("Validation loss: {}".format(val_loss))

        current_bal_acc = val_metrics['balanced_accuracy']
        print(
            f"Balanced Acc: {current_bal_acc:.4f} | "
            f"ROC-AUC: {val_metrics['roc_auc']:.4f} | "
            f"Avg Precision: {val_metrics['avg_precision']:.4f}"
        )

        # Save model only if it's better than the previous best
        if current_bal_acc > best_bal_acc:
            print(f"✓ New best! Previous: {best_bal_acc:.4f} → Current: {current_bal_acc:.4f}")
            best_bal_acc = current_bal_acc

            os.makedirs(args.output_dir, exist_ok=True)

            # Save model checkpoint
            torch.save(model.state_dict(), best_model_path)
            print(f"  Model saved at {best_model_path}")
        else:
            print(f"  No improvement (Best: {best_bal_acc:.4f})")

    print(f"\n Training completed! Best Balanced Accuracy: {best_bal_acc:.4f}")

    plot_metrics(
        args.output_dir,
        fold,
        train_losses, val_losses,
        train_bal_accs, val_bal_accs,
        train_roc_aucs, val_roc_aucs,
        train_avg_precs, val_avg_precs
    )


def validation_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0


    for batch in val_loader:
        inputs = batch['image']
        targets = batch['label']

        outputs = model(inputs)
        loss = criterion(outputs, targets)


        val_loss += loss.item()

    return val_loss / len(val_loader)


