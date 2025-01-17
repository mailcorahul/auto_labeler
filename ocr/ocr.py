import os
import cv2
import json
import numpy as np

from PIL import Image
from tqdm import tqdm

import torch
from doctr.io import DocumentFile

from config import OCR_CONFIG
from model import Models

class OCR:
    def __init__(self, unlabelled_dump_path, result_path):

        self.unlabelled_dump_path = unlabelled_dump_path
        self.unlabelled_image_paths, self.pseudo_labels = [], {}
        self.result_path = result_path

        self.model_data = Models().create_model()

    def label_images_for_ocr_task(self):
        """Label unlabelled data using OCR Models."""

        files = os.listdir(self.unlabelled_dump_path)
        for file in files:
            self.unlabelled_image_paths.append(os.path.join(self.unlabelled_dump_path, file))

        model, processor, prompt = self.model_data["model"], self.model_data["processor"], self.model_data["prompt"]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        max_new_tokens = 100

        for image_path in tqdm(self.unlabelled_image_paths):
            image = Image.open(image_path).convert("RGB")
            if OCR_CONFIG["model"] == "mindee/doctr":
                doc = DocumentFile.from_images(image_path)
                result = model(doc)
                self.pseudo_labels[image_path] = result.export()

            else:
                pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
                generated_ids = model.generate(pixel_values)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                self.pseudo_labels[image_path] = generated_text

        with open(self.result_path, "w") as f:
            json.dump(self.pseudo_labels, f, indent=2)


    def label_images(self):
        """Labels image classification data based on the inputs config params."""

        print('[/] labeling images...')
        self.label_images_for_ocr_task()