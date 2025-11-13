from args import get_args
import torch
import os
import pandas as pd
from dataset import KneeXrayDataset
from torch.utils.data import DataLoader
from model import MyModel
from trainer import train_model
import gc


# device setup
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using device: MPS")
else:
    device = torch.device("cpu")
    print(f"Using device: CPU")


def main():
    # 1. We need some arguments
    args = get_args()

    # 2. Iterate among the folds
    for fold in range(1, 6):
        print("=" * 60)
        print('Training fold: ', fold)

        train_set = pd.read_csv(os.path.join(args.csv_dir, 'fold_{}_train.csv'.format(str(fold))))
        val_set = pd.read_csv(os.path.join(args.csv_dir, 'fold_{}_val.csv'.format(str(fold))))

        # 3. Preparing datasets
        train_dataset = KneeXrayDataset(train_set, cache=False)
        val_dataset = KneeXrayDataset(val_set, cache=False)

        # 4. Creating data loaders
        num_workers = 4
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            prefetch_factor=1,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            prefetch_factor=1,
            persistent_workers=True,
        )

        # 5. Initializing the model
        model = MyModel(args.backbone).to(device)

        # 6. Train the model
        train_model(model, train_loader, val_loader, fold)

        # cleanup per fold
        del train_loader, val_loader, train_dataset, val_dataset, model
        gc.collect()
        torch.cuda.empty_cache()

    print("All folds finished.")


if __name__ == '__main__':
    main()