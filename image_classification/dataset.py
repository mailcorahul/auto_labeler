import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

class DatasetPretrained(Dataset):

    def __init__(self, image_paths, preprocess=None):
        self.image_paths = image_paths

        # setting the transforms
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        if preprocess is not None:
            self.transform = preprocess
        else:
            self.transform =  transforms.Compose([
                transforms.Resize(224),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """ Read and transform image """
        image = self.transform(Image.open(self.image_paths[idx]))
        return image