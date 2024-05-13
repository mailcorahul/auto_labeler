import os
import cv2
import json
import numpy as np

from PIL import Image
from tqdm import tqdm

from config import IMAGE_CLASSIFIER_CONFIG
from model import Models

class VisionLanguageModel:
    def __init__(self, unlabelled_dump_path, result_path):

        self.unlabelled_dump_path = unlabelled_dump_path
        self.unlabelled_image_paths, self.pseudo_labels = [], []
        self.result_dir = result_path

        self.model_data = Models().create_model()

    def label_images_using_vlm(self):
        """Label unlabelled data using Vision Language Models."""

        files = os.listdir(self.unlabelled_dump_path)
        for file in files:
            self.unlabelled_image_paths.append(os.path.join(self.unlabelled_dump_path, file))

        model, processor, prompt = self.model_data["model"], self.model_data["processor"], self.model_data["prompt"]
        for image_path in self.unlabelled_image_paths:
            image = Image.open(image_path)
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            generate_ids = model.generate(**inputs, max_length=30)
            generated_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            print(generated_text)


    def generate_labelled_data(self):
        """Create a new dump of classification labelled data."""

        # create result dir with subfolders for every class.
        for class_label in self.classes:
            os.makedirs(os.path.join(self.result_dir, class_label), exist_ok=True)

        # copy images to labelled directory
        for pseudo_label, image_path in zip(self.pseudo_labels, self.unlabelled_image_paths):
            dest_path = os.path.join(self.result_dir, pseudo_label)
            os.system(f'cp {image_path} {dest_path}')


    def label_images(self):
        """Labels image classification data based on the inputs config params."""

        self.label_images_using_vlm()
        self.generate_labelled_data()