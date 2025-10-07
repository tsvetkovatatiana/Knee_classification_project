# import os
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# metadata = pd.DataFrame(columns=["Name", "Path", "KL"])

# main_dir = "data/archive/OSAIL_KL_Dataset/Labeled"
#
# for grade in range(5):
#     folder_path = os.path.join(main_dir, str(grade))
#     print(folder_path)
#     for image in os.listdir(folder_path):
#         image_path = os.path.join(folder_path, image)
#
#         metadata = metadata._append({
#             "Name" : image,
#             "Path" : image_path,
#             "KL" : grade
#         }, ignore_index = True)
#
# metadata.to_csv("metadata.csv", index=False)

metadata = pd.read_csv("metadata.csv")

kl_grades_distribution = sns.countplot(data=metadata, x="KL", palette='viridis', alpha=0.8)

plt.title('Distribution of KL Grades of the whole dataset', fontsize=16, fontweight='bold')
plt.xlabel('KL Grade', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.savefig('Distribution_of_KL_grades_whole_dataset.png', dpi=300, bbox_inches='tight')
plt.show()

print("KL distribution --------------")
print(metadata["KL"].value_counts())

train_val_data, test_data = train_test_split(metadata, test_size=0.2, stratify=metadata["KL"])
train_data, val_data = train_test_split(
    train_val_data,
    test_size=0.20,
    stratify=train_val_data["KL"],
    random_state=42
)

kl_grades_distr_train_data = sns.countplot(data=train_data, x="KL", palette='rocket', alpha=0.8)

plt.title('Distribution of KL Grades. Training data', fontsize=16, fontweight='bold')
plt.xlabel('KL Grade', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.savefig('Distribution_of_KL_grades_Training_data.png', dpi=300, bbox_inches='tight')
plt.show()

kl_grades_distribution_val_data = sns.countplot(data=val_data, x="KL", palette='viridis', alpha=0.8)

plt.title('Distribution of KL Grades. Validation data', fontsize=16, fontweight='bold')
plt.xlabel('KL Grade', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.savefig('Distribution_of_KL_grades_Validation_data.png', dpi=300, bbox_inches='tight')
plt.show()

kl_grades_distribution_test_data = sns.countplot(data=val_data, x="KL", palette='rocket', alpha=0.8)

plt.title('Distribution of KL Grades. Test data', fontsize=16, fontweight='bold')
plt.xlabel('KL Grade', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.savefig('Distribution_of_KL_grades_Test_data.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

