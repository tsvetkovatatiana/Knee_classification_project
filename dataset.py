from torch.utils.data import Dataset
import cv2
import numpy as np
from transforms import *


def read_x_ray(path):
    x_ray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    x_ray = x_ray.astype(np.float32) / 255

    x_ray_3ch = np.zeros((3, x_ray.shape[0], x_ray.shape[1]), dtype=x_ray.dtype)  # shape is (3, H, W)
    # to have a 3d image we assign to each channel
    x_ray_3ch[0] = x_ray
    x_ray_3ch[1] = x_ray
    x_ray_3ch[2] = x_ray

    return x_ray_3ch


class KneeXrayDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img = read_x_ray(self.dataset["Path"].iloc[index])
        label = self.dataset["KL"].iloc[index]

        # if self.transforms is not None:
        #     for t in self.transforms:
        #         if hasattr(t, "randomize"):



        results = {
            "img": img,
            "label": label
        }
        return results
