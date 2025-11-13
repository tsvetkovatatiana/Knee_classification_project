from torch.utils.data import Dataset
import cv2
import numpy as np
from transform_simple import *


def read_x_ray(path):
    x_ray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x_ray = x_ray.astype(np.float32) / 255.0

    x_ray_3ch = np.zeros((3, x_ray.shape[0], x_ray.shape[1]), dtype=x_ray.dtype)  # shape is (3, H, W)
    # to have a 3d image we assign to each channel
    x_ray_3ch[0] = x_ray
    x_ray_3ch[1] = x_ray
    x_ray_3ch[2] = x_ray

    return x_ray_3ch


class KneeXrayDataset(Dataset):
    def __init__(self, dataframe, train=True, cache=False):
        self.data = dataframe.reset_index(drop=True)
        self.train = train
        self.cache = cache

        self.transform = train_transform if train else val_transform

        self.cached_images = {}
        if self.cache:
            print("[INFO] Caching images in memory...")
            for i in range(len(self.data)):
                path = self.data["Path"].iloc[i]
                self.cached_images[path] = read_x_ray(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data["Path"].iloc[index]
        label = int(self.data["KL"].iloc[index])

        # Load from cache or disk
        if self.cache and path in self.cached_images:
            image = self.cached_images[path]
        else:
            image = read_x_ray(path)

        # Apply augmentations
        if self.transform:
            image = self.transform(image=image)["image"]

        return {"img": image, "label": label}