from args import get_args
from datasets import KneeXrayDataset
import pandas as pd
import os
from torch.utils.data import DataLoader
from models import MyModel
from trainer import train_model


def main():
    # 1. We need some arguments
    args = get_args()

    # 2. Iterate among folds
    print("Training started...-----------------------------")
    for fold in range(1, 6):
        print(f"Training fold: {fold}")

        train_set = pd.read_csv(os.path.join(args.csv_dir, "fold_{}_train.csv".format(str(fold))))
        val_set = pd.read_csv(os.path.join(args.csv_dir, "fold_{}_val.csv".format(str(fold))))

        # 3. Preparing datasets
        train_dataset = KneeXrayDataset(train_set)
        val_dataset = KneeXrayDataset(val_set)

        # 4. Create data loader
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # 5. Initializing the model
        model = MyModel(args.backbone)

        # 6. Train the model
        train_model(model, train_loader, val_loader, fold)


if __name__ == "__main__":
    main()
