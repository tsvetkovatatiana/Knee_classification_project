import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Model training options")

    parser.add_argument("-backbone", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50"])

    parser.add_argument("-csv_dir", type=str, default="data/CSVs")

    parser.add_argument("-batch_size", type=int, default=32,
                        choices=[16, 32, 64])

    parser.add_argument("-output_dir", type=str, default="session")

    parser.add_argument("-lr", type=float, default=1e-3)

    parser.add_argument("-epochs", type=float, default=5)

    args = parser.parse_args()

    return args
