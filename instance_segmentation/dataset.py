import cv2
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# specific to SAM
def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device)
    return image.permute(2, 0, 1).contiguous()

class SegmentationDataset(Dataset):

    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """ Read and transform image """
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        return idx, image


class SegmentationCustomDataset:

    def __init__(self, image_paths, batch_size):

        self.image_paths = image_paths
        self.batch_size = batch_size
        self.batched_image_paths = self.create_batches()

    def create_batches(self):

        batched_image_paths = []
        i = 0

        while i < len(self.image_paths):
            batch = self.image_paths[i: i + self.batch_size]
            batched_image_paths.append(batch)
            i += self.batch_size

        return batched_image_paths

    # def generate_batch(self):

    #     for image_paths in self.batched_image_paths:
    #         batch_images = []
    #         for image_path in image_paths:
    #             image = Image.open(image_path).convert("RGB")
    #             batch_images.append(image)

    #         yield batch_images, image_paths