import os
import cv2
import json
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import SegmentationCustomDataset, SegmentationDataset

from visualization import show_masks_on_image

from config import INSTANCE_SEGMENTOR_CONFIG
from model import Models


class InstanceSegmentationLabeler:

    def __init__(self, unlabelled_dump_path, class_texts_path, result_path, reference_images_path, viz=False, viz_path=None):

        self.unlabelled_dump_path = unlabelled_dump_path
        self.class_texts_path = class_texts_path
        self.result_path = result_path
        self.reference_images_path = reference_images_path

        self.model_data = Models().create_model()
        self.model, self.processor = self.model_data["model"], self.model_data["processor"]
        self.label_mode = INSTANCE_SEGMENTOR_CONFIG["auto_label_mode"]

        self.unlabelled_image_paths = []

        self.class_texts = []
        if self.label_mode != "zero-shot":
            with open(self.class_texts_path) as f:
                self.class_texts = json.load(f)

        self.viz = viz
        self.viz_path = viz_path
        if self.viz:
            os.makedirs(self.viz_path, exist_ok=True)

    def label_images(self):

        # get all raw image file paths
        files = os.listdir(self.unlabelled_dump_path)
        for file in files:
            self.unlabelled_image_paths.append(os.path.join(self.unlabelled_dump_path, file))

        # image loader and transforms
        dataset = SegmentationCustomDataset(
            image_paths=self.unlabelled_image_paths,
            batch_size=self.model_data["batch_size"]
        )
        dataloader = DataLoader(dataset, batch_size=self.model_data["batch_size"], shuffle=False)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = self.model_data["model"]

        # resize_transform = ResizeLongestSide(model.image_encoder.img_size)

        imagepath2labels = {}
        # generate embeddings for all classes
        with torch.no_grad():#, torch.cuda.amp.autocast():
            for batched_image_paths in tqdm(dataset.batched_image_paths):

                batch_images = []
                for image_path in batched_image_paths:
                    image = Image.open(image_path).convert("RGB")
                    batch_images.append(image)

                # outputs = model(batch_images_transformed, multimask_output=True)
                outputs = model(batch_images, points_per_batch=64)
                # image_path = self.unlabelled_image_paths[item_idx]
                for image_path, output in zip(batched_image_paths, outputs):
                    imagepath2labels[image_path] = {
                        "masks": output["masks"],
                        "scores": output["scores"].cpu().numpy()
                    }

                    # print(output)

                if self.viz:
                    for image, output in zip(batch_images, outputs):
                        masks = output["masks"]
                        h, w = masks[0].shape[-2:]
                        mask_image = (np.ones((h, w, 3)) * 255)
                        for mask in masks:
                            color = np.concatenate([np.random.random(3)]) * 255#, np.array([0.6])], axis=0)
                            h, w = mask.shape[-2:]
                            mask_image = mask_image * (mask.reshape(h, w, 1) * color.reshape(1, 1, -1))

                        cv2.imwrite("/tmp/sam.png", mask_image)
                        # show_masks_on_image(image, masks)


        # with open(self.result_path, "w") as f:
        #     json.dump(imagepath2labels, f, indent=2)