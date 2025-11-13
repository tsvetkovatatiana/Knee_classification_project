import json
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    cohen_kappa_score
)
# from sklearn.preprocessing import label_binarize


def calculate_metrics(y_true, y_pred, y_probs):
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: True labels (list or numpy array)
        y_pred: Predicted labels (list or numpy array)
        y_probs: Prediction probabilities (numpy array of shape [n_samples, n_classes])

    Returns:
        dict: Dictionary containing all calculated metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {'accuracy': accuracy_score(y_true, y_pred),  # Basic metrics
               'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
               'f1_score': f1_score(y_true, y_pred, average='macro', zero_division=0),  # Multi-class with macro avg
               'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
               'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
               'confusion_matrix': confusion_matrix(y_true, y_pred), 'cohens_kappa': cohen_kappa_score(y_true, y_pred)}

    # ROC-AUC (multi-class)
    try:
        num_classes = y_probs.shape[1]
        if num_classes == 2:
            # Binary classification
            metrics['roc_auc'] = roc_auc_score(y_true, y_probs[:, 1])
        else:
            # Multi-class: use one-vs-rest with macro averaging
            metrics['roc_auc'] = roc_auc_score(
                y_true, y_probs, multi_class='ovr', average='macro'
            )
    except (ValueError, IndexError) as e:
        print(f"Warning: Could not calculate ROC-AUC: {e}")
        metrics['roc_auc'] = float('nan')

    return metrics


def plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, fold, save_dir):
    """
    Plot and save training curves for all metrics.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_metrics: List of dictionaries containing training metrics per epoch
        val_metrics: List of dictionaries containing validation metrics per epoch
        fold: Fold number
        save_dir: Directory to save plots
    """
    fold_dir = os.path.join(save_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    # Extract metrics from history
    metric_names = ['accuracy', 'balanced_accuracy', 'f1_score', 'precision',
                    'recall', 'roc_auc', 'cohens_kappa']

    # Extract metric values for each epoch
    train_metric_values = {}
    val_metric_values = {}

    for metric_name in metric_names:
        train_metric_values[metric_name] = [epoch_metrics[metric_name] for epoch_metrics in train_metrics]
        val_metric_values[metric_name] = [epoch_metrics[metric_name] for epoch_metrics in val_metrics]

    # Plot each metric
    metric_display_names = {
        'accuracy': 'Accuracy',
        'balanced_accuracy': 'Balanced Accuracy',
        'f1_score': 'F1 Score',
        'precision': 'Precision',
        'recall': 'Recall',
        'roc_auc': 'ROC-AUC',
        'cohens_kappa': "Cohen's Kappa"
    }

    for metric_name in metric_names:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_metric_values[metric_name], 'b-o',
                 label=f'Train {metric_display_names[metric_name]}', linewidth=2, markersize=4)
        plt.plot(epochs, val_metric_values[metric_name], 'r-s',
                 label=f'Validation {metric_display_names[metric_name]}', linewidth=2, markersize=4)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(metric_display_names[metric_name], fontsize=12)
        plt.title(f'{metric_display_names[metric_name]} over Epochs - Fold {fold}',
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(fold_dir, f'{metric_name}.png'), dpi=150)
        plt.close()

    # Plot confusion matrix for the last epoch
    plot_confusion_matrix(
        val_metrics[-1]['confusion_matrix'],
        fold=fold,
        save_dir=fold_dir
    )

    print(f"Training curves and confusion matrix saved in {fold_dir}/")


def plot_confusion_matrix(cm, fold, save_dir):
    """
    Plot and save confusion matrix.

    Args:
        cm: Confusion matrix (numpy array)
        fold: Fold number
        save_dir: Directory to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.colormaps['viridis'])
    ax.figure.colorbar(im, ax=ax)

    # Add labels
    n_classes = cm.shape[0]
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=np.arange(n_classes),
           yticklabels=np.arange(n_classes),
           title=f'Confusion Matrix - Fold {fold}',
           ylabel='True Label',
           xlabel='Predicted Label')

    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()


def save_checkpoint(cur_fold, epoch, model, y_true, y_pred, val_metric,
                    best_val_metric=None, prev_model_path=None,
                    comparator="gt", save_dir="session"):
    """
    Save the best model checkpoint for each fold based on validation metric.
    Also saves metrics.json with metadata.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Determine if current metric is better
    improved = False
    if best_val_metric is None:
        improved = True
    elif comparator == "gt" and val_metric > best_val_metric:
        improved = True
    elif comparator == "lt" and val_metric < best_val_metric:
        improved = True

    if improved:
        # Remove old checkpoint (only one per fold)
        if prev_model_path and os.path.exists(prev_model_path):
            try:
                os.remove(prev_model_path)
            except OSError:
                pass

        # Define new file paths
        ckpt_path = os.path.join(save_dir, f"fold_{cur_fold}_best.pth")
        metrics_path = os.path.join(save_dir, f"fold_{cur_fold}_metrics.json")

        # Save model
        torch.save(model.state_dict(), ckpt_path)

        # Save metadata
        metrics_data = {
            "fold": cur_fold,
            "best_epoch": epoch + 1,
            "best_val_metric": float(val_metric),
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=4)

        print(f"✅ Best model updated — fold {cur_fold}, epoch {epoch+1}, metric={val_metric:.4f}")
        print(f"Saved model to: {ckpt_path}")
        return val_metric, ckpt_path

    # Not improved — keep old best
    return best_val_metric, prev_model_path



def plot_loss_curve(train_losses, val_losses, save_dir, fold):
    """
        Plot the training and validation loss curves.

        Args:
            train_losses (list): List of average training losses per epoch.
            val_losses (list): List of average validation losses per epoch.
            save_dir (str): Directory to save the plot image.
            fold (int): Current fold number for naming the output file.
        """
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)

    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    filename = f"loss_curve_fold_{fold}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    print(f"[INFO] Saved training curve to {save_path}")
    plt.close()
