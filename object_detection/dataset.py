import cv2
from PIL import Image

import numpy as np
import torch
from torchvision import transforms

class DetectionDataset:

    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """ Read and transform image """
        # image = torch.Tensor(cv2.imread(self.image_paths[idx]))
        pil_image = torch.Tensor(np.array(Image.open(self.image_paths[idx])))
        # print(pil_image.size())
        return idx, pil_image
