import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Model training options")

    parser.add_argument("-backbone", type=str, default="resnet34",
                        choices=["resnet18", "resnet34", "resnet50"])

    parser.add_argument("-csv_dir", type=str, default="data/CSVs")

    parser.add_argument("-batch_size", type=int, default=32,
                        choices=[16, 32, 64])

    parser.add_argument("-output_dir", type=str, default="session")

    parser.add_argument("-lr", type=float, default=1e-3)

    parser.add_argument("-epochs", type=float, default=20)

    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument("-scheduler", type=str, default="none",
                        choices=["cosine", "plateau", "step", "none"],
                        help="Learning rate scheduler type")
    parser.add_argument("-min_lr", type=float, default=1e-3,
                        help="Minimum LR for cosine scheduler")
    parser.add_argument("-step_size", type=int, default=10,
                        help="Step size for StepLR (if used)")
    parser.add_argument("-gamma", type=float, default=0.5,
                        help="Gamma decay factor for StepLR/Plateau")

    args = parser.parse_args()

    return args
