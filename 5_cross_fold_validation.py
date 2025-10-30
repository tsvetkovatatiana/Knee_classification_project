import os
from os import path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns


metadata = pd.read_csv("metadata.csv")

os.makedirs("data/CSVs", exist_ok=True)


train_val_data, test_data = train_test_split(metadata, test_size=0.2, stratify=metadata["KL"])

# Save test data
test_data.to_csv("data/CSVs/test_data.csv", index=False)
print(f"Test data saved: {len(test_data)} samples")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_train_val = train_val_data.drop('KL', axis=1)  # Features
y_train_val = train_val_data['KL']  # Target

for fold, (train_index, val_index) in enumerate(skf.split(X_train_val, y_train_val), 1):
    train_fold = train_val_data.iloc[train_index]
    val_fold = train_val_data.iloc[val_index]

    # Save to CSV files
    train_fold.to_csv(f"data/CSVs/fold_{fold}_train.csv", index=False)
    val_fold.to_csv(f"data/CSVs/fold_{fold}_val.csv", index=False)

csv_dir = "data/CSVs"


def plot_distribution(data, x, plot_name):
    plt.figure(figsize=(10, 6))
    plot = sns.countplot(data, x=x, palette='viridis', alpha=0.8)
    plt.title(f'Distribution of KL Grades. {plot_name}', fontsize=16, fontweight='bold')
    plt.xlabel('KL Grade', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.savefig(f'{plot_name}_distribution.png', dpi=300, bbox_inches='tight')
    return plot


# Bar charts
for fold_file in os.listdir(csv_dir):
    fold_name, extension = path.splitext(fold_file)
    fold_data = pd.read_csv(os.path.join(csv_dir, fold_file))
    kl_grades_distr = plot_distribution(fold_data, "KL", fold_name)
    plt.show()
    plt.close()




