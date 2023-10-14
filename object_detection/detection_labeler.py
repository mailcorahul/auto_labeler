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

from dataset import DetectionDataset

from config import OBJECT_DETECTOR_CONFIG
from model import Models


class DetectionLabeler:

    def __init__(self, unlabelled_dump_path, class_texts_path, result_path, reference_images_path, viz=False, viz_path=None):

        self.unlabelled_dump_path = unlabelled_dump_path
        self.class_texts_path = class_texts_path
        self.result_path = result_path
        self.reference_images_path = reference_images_path

        self.model_data = Models().create_model()
        self.model, self.processor = self.model_data["model"], self.model_data["processor"]
        self.label_mode = OBJECT_DETECTOR_CONFIG["auto_label_mode"]

        self.unlabelled_image_paths = []

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
        dataset = DetectionDataset(image_paths=self.unlabelled_image_paths)
        dataloader = DataLoader(dataset, batch_size=self.model_data["batch_size"], shuffle=False)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = self.model_data["model"]
        model = model.to(device)

        imagepath2labels = {}
        # generate embeddings for all classes
        with torch.no_grad():#, torch.cuda.amp.autocast():
            for batch_idx, batch_data in enumerate(tqdm(dataloader)):
                #print('[/] batch idx: {}, forward pass...'.format(batch_idx))
                item_idx, batch_images = batch_data
                inputs = self.processor(
                    text=self.class_texts, images=batch_images, return_tensors="pt"
                    )
                inputs = inputs.to(device)
                # for k,v in inputs.items():
                #   print(k,v.shape)
                outputs = model(**inputs)

                # Convert outputs (bounding boxes and class logits) to COCO API
                # print(batch_images[0].size()[:2])
                target_sizes = torch.Tensor([batch_images[0].size()[:2]]).to(device)
                results = self.processor.post_process_object_detection(
                            outputs=outputs,
                            target_sizes=target_sizes,
                            threshold=0.2
                        )

                # rearrange the detections and dump it to results path
                image_path = self.unlabelled_image_paths[item_idx]
                imagepath2labels[image_path] = []
                for result in results:
                    boxes, scores, labels = result["boxes"], result["scores"], result["labels"]
                    for box, score, label in zip(boxes, scores, labels):
                        box, score, label = box.tolist(), score.tolist(), label.tolist()
                        box = [round(i, 2) for i in box]
                        detection = {
                            "box": box,
                            "score": round(score, 4),
                            "label": self.class_texts[0][label]
                        }

                        # print(type(box), type(score), type(label))
                        imagepath2labels[image_path].append(detection)

                if self.viz:
                    for result in results:
                        boxes, scores, labels = result["boxes"], result["scores"], result["labels"]
                        # print(boxes, scores, labels)

                        image = Image.open(image_path)
                        draw = ImageDraw.Draw(image)

                        for box, score, label in zip(boxes, scores, labels):
                            box = [round(i, 2) for i in box.tolist()]
                            x1, y1, x2, y2 = tuple(box)
                            draw.rectangle(xy=((x1, y1), (x2, y2)), outline="red")
                            draw.text(xy=(x1, y1), text=self.class_texts[0][label])

                        filename = os.path.split(image_path)[1]
                        image.save(os.path.join(self.viz_path, filename))

        with open(self.result_path, "w") as f:
            json.dump(imagepath2labels, f, indent=2)