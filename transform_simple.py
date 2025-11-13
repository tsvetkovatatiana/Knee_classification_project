import albumentations as A
from albumentations.pytorch import ToTensorV2


# Train augmentations — gentle for X-rays
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5, border_mode=0
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.1, contrast_limit=0.1, p=0.3
    ),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Validation — only resize + normalize
val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
