from transforms import *
from args import get_args
import torch
import os
import pandas as pd
from dataset import KneeXrayDataset
from torch.utils.data import DataLoader
from model import MyModel
from trainer import train_model

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

def main():
    # 1. We need some arguments
    args = get_args()

    # 2. Iterate among the folds
    for fold in range(1, 6):
        print('Training fold: ', fold)

        train_set = pd.read_csv(os.path.join(args.csv_dir, 'fold_{}_train.csv'.format(str(fold))))
        val_set = pd.read_csv(os.path.join(args.csv_dir, 'fold_{}_val.csv'.format(str(fold))))

        # 3. Preparing datasets
        train_dataset = KneeXrayDataset(
            train_set)
        val_dataset = KneeXrayDataset(val_set)

        # 4. Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=torch.mps.is_available())
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=torch.mps.is_available())

        # 5. Initializing the model
        model = MyModel(args.backbone).to(device)

        # 6. Train the model
        train_model(model, train_loader, val_loader, fold)


if __name__ == '__main__':
    main()