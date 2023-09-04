import os
import cv2
import json
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

from tqdm import tqdm

from config import IMAGE_CLASSIFIER_CONFIG
from model import Models
from dataset import DatasetPretrained

class Text2ImageRetrieval:
    def __init__(self, class_prompts_path, unlabelled_dump_path, result_path):

        self.class2captions = self.load_class_prompts(class_prompts_path)

        self.classes, self.captions = [], []
        for class_, caption in self.class2captions.items():
            self.classes.append(class_)
            self.captions.append(caption)

        self.unlabelled_dump_path = unlabelled_dump_path
        self.unlabelled_image_paths, self.pseudo_labels = [], []
        self.result_dir = result_path

        self.model_data = Models().create_model()

    def load_class_prompts(self, class_prompts_path):
        """Collect list of prompts describing every class in the new dataset."""

        with open(class_prompts_path) as f:
            class2captions = json.load(f)

        return class2captions

    def retrieve_images_using_text(self):
        """Retrieve images with text using text-to-tmage architecture such as CLIP/BLIP/BeIT."""

        model, preprocess, tokenizer = None, None, None
        if IMAGE_CLASSIFIER_CONFIG["model"] == "clip":
            model, preprocess, tokenizer = self.model_data["model"], self.model_data["preprocess"], self.model_data["tokenizer"]

        files = os.listdir(self.unlabelled_dump_path)
        for file in files:
            self.unlabelled_image_paths.append(os.path.join(self.unlabelled_dump_path, file))

        # image loader and transforms
        dataset = DatasetPretrained(
            image_paths=self.unlabelled_image_paths,
            preprocess=preprocess
        )
        dataloader = DataLoader(dataset, batch_size=self.model_data["batch_size"], shuffle=False)
        text = tokenizer(self.captions)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        text = text.to(device)

        # generate embeddings for all classes
        class_indices = []
        with torch.no_grad():#, torch.cuda.amp.autocast():
            for batch_idx, batch_images in enumerate(tqdm(dataloader)):
                batch_images = batch_images.to(device)
                #print('[/] batch idx: {}, forward pass...'.format(batch_idx))
                image_features = model.encode_image(batch_images)
                text_features = model.encode_text(text)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

                max_prob_indices = torch.argmax(text_probs, dim=1).tolist()
                class_indices += max_prob_indices

        # convert class idx to class label
        for idx in class_indices:
            self.pseudo_labels.append(self.classes[idx])

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

        self.retrieve_images_using_text()
        self.generate_labelled_data()