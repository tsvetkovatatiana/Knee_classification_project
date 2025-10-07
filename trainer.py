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

    with torch.no_grad():
        for batch in val_loader:
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

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()

    return val_loss / len(val_loader)


def calculate_val_accuracy_metrics(model, val_loader, device):
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['img']
            labels = batch['label']

            if not isinstance(inputs, torch.Tensor):
                inputs = torch.tensor(inputs, dtype=torch.float32)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long)

            # Ensure labels are 1D
            if labels.dim() > 1:
                labels = labels.squeeze()

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            # Get probabilities for all classes
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Concatenate all probabilities
    all_probs = np.vstack(all_probs)

    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)

    # For multi-class: use one-vs-rest ROC-AUC
    try:
        num_classes = all_probs.shape[1]
        if num_classes == 2:
            # Binary classification: use probability of positive class
            roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            # Multi-class: use ovr (one-vs-rest) with macro averaging
            roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except (ValueError, IndexError):
        roc_auc = float('nan')

    # For multi-class: use macro averaging for average precision
    try:
        if num_classes == 2:
            avg_precision = average_precision_score(all_labels, all_probs[:, 1])
        else:
            # Binarize labels for multi-class average precision
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(all_labels, classes=range(num_classes))
            avg_precision = average_precision_score(y_bin, all_probs, average='macro')
    except (ValueError, IndexError):
        avg_precision = float('nan')

    return {
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": roc_auc,
        "avg_precision": avg_precision
    }


def calculate_train_accuracy_metrics(all_labels, all_preds, all_probs):
    # Ensure all_probs is a numpy array
    if not isinstance(all_probs, np.ndarray):
        all_probs = np.array(all_probs)

    # Handle case where probs might be a list of arrays
    if all_probs.ndim == 1 or (all_probs.ndim == 3):
        raise ValueError(f"Unexpected all_probs shape: {all_probs.shape}. Expected 2D array.")

    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)

    num_classes = all_probs.shape[1]

    try:
        if num_classes == 2:
            roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except (ValueError, IndexError):
        roc_auc = float('nan')

    try:
        if num_classes == 2:
            avg_precision = average_precision_score(all_labels, all_probs[:, 1])
        else:
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(all_labels, classes=range(num_classes))
            avg_precision = average_precision_score(y_bin, all_probs, average='macro')
    except (ValueError, IndexError):
        avg_precision = float('nan')

    return {
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": roc_auc,
        "avg_precision": avg_precision
    }


def plot_metrics(out_dir, fold, train_losses, val_losses,
                 train_bal_accs, val_bal_accs,
                 train_roc_aucs, val_roc_aucs,
                 train_avg_precs, val_avg_precs):
    # Create a separate folder for each fold
    fold_dir = os.path.join(out_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    def save_plot(train_values, val_values, title, ylabel, filename):
        plt.figure()
        plt.plot(epochs, train_values, label='Train')
        plt.plot(epochs, val_values, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(f'{title} - Fold {fold}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(fold_dir, f"{filename}.png"))
        plt.close()

    save_plot(train_losses, val_losses, 'Loss over Epochs', 'Loss', 'loss')
    save_plot(train_bal_accs, val_bal_accs, 'Balanced Accuracy over Epochs', 'Balanced Accuracy', 'balanced_accuracy')
    save_plot(train_roc_aucs, val_roc_aucs, 'ROC-AUC over Epochs', 'ROC-AUC', 'roc_auc')
    save_plot(train_avg_precs, val_avg_precs, 'Average Precision over Epochs', 'Average Precision', 'avg_precision')

    print(f"📊 Training curves for fold {fold} saved in {fold_dir}/ as PNG files.")