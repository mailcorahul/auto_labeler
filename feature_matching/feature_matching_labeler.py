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

from dataset import FeatureMatchingDataset

from config import FEATURE_MATCHING_CONFIG
from model import Models


class FeatureMatchingLabeler:

    def __init__(self, unlabelled_dump_path, reference_images, result_path, viz=False, viz_path=None):

        self.unlabelled_dump_path = unlabelled_dump_path
        self.reference_images = reference_images
        self.result_path = result_path

        self.model_data = Models().create_model()
        self.model, self.processor = self.model_data["model"], self.model_data["processor"]
        self.label_mode = FEATURE_MATCHING_CONFIG["auto_label_mode"]

        self.unlabelled_image_paths = []
        self.reference_image_paths = []

        self.viz = viz
        self.viz_path = viz_path
        if self.viz:
            os.makedirs(self.viz_path, exist_ok=True)

    def label_images(self):

        # get all raw image file paths
        files = os.listdir(self.unlabelled_dump_path)
        for file in files:
            self.unlabelled_image_paths.append(os.path.join(self.unlabelled_dump_path, file))

        # get all reference image file paths
        files = os.listdir(self.reference_images)
        for file in files:
            self.reference_image_paths.append(os.path.join(self.reference_images, file))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = self.model_data["model"]
        model = model.to(device)

        # match every reference image with every unlabeled image and get the positive matches
        if FEATURE_MATCHING_CONFIG["model"] == "superglue":
            for ref_idx, ref_image_path in enumerate(self.reference_image_paths):
                ref_image = Image.open(ref_image_path)
                for image_idx, image_path in enumerate(self.unlabelled_image_paths):
                    image_to_label = Image.open(image_path)
                    images = [ref_image, image_to_label]

                    inputs = processor(images, return_tensors="pt")
                    with torch.no_grad():#, torch.cuda.amp.autocast():
                        outputs = model(**inputs)

                        image_sizes = [(image.height, image.width) for image in images]
                        outputs = processor.post_process_keypoint_matching(
                            outputs,
                            image_sizes,
                            threshold=0.0
                        )
