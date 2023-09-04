import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

from dataset import DatasetPretrained

class ClassificationLabeler:

    def __init__(
        self,
        class_txt_path,
        reference_dump_path,
        unlabelled_dump_path,
        result_path
        ):

        self.classes = self.create_class_names(class_txt_path)
        self.reference_dump_path = reference_dump_path
        self.unlabelled_dump_path = unlabelled_dump_path
        self.result_dir = result_path
        self.unlabelled_image_paths = []

        self.class_reference_embeddings, self.unlabelled_image_embeddings = [], []
        self.pseudo_labels = []

        self.model = self.create_backbone(architecture='resnet50')
        self.batch_size = 128
        self.distance_metric = 'euclidean'


    def create_class_names(self, class_txt_path):
        """Create a list of class names to label."""

        with open(class_txt_path) as f:
            classes = f.read().split('\n')
            classes = list(filter(None, classes))

        return classes

    def create_backbone(self, architecture='resnet50'):
        """Create a pretrained model object for labelling."""

        backbone = getattr(models, architecture)
        print('[/] {}'.format(backbone))
        backbone = backbone(pretrained=True)
        backbone = nn.Sequential(*list(backbone.children())[:-1], nn.AdaptiveAvgPool2d(1))
        backbone.cuda()
        backbone.eval()

        return backbone

    def create_reference_embeddings(self):
        """Create reference embeddings for every class."""

        # load reference images for all classes
        reference_image_paths, class_indices = [], []
        for class_idx, class_label in enumerate(self.classes):
            files = os.listdir(os.path.join(self.reference_dump_path, class_label))
            for file in files:
                image_path = os.path.join(self.reference_dump_path, class_label, file)
                reference_image_paths.append(image_path)
                class_indices.append(class_idx)

        dataset = DatasetPretrained(image_paths=reference_image_paths)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # generate embeddings for all classes
        for batch_idx, batch_images in enumerate(dataloader):
            batch_images = batch_images.cuda()
            #print('[/] batch idx: {}, forward pass...'.format(batch_idx))
            batch_embeddings = self.model(batch_images).cpu().detach().numpy()
            self.class_reference_embeddings += batch_embeddings

        return class_indices

    def create_embeddings_for_unlabelled_images(self):
        """Create representations for unlabelled images."""

        files = os.listdir(self.unlabelled_dump_path)
        for file in files:
            self.unlabelled_image_paths.append(os.path.join(self.unlabelled_dump_path, file))

        dataset = DatasetPretrained(image_paths=self.unlabelled_image_paths)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # generate embeddings for all classes
        for batch_idx, batch_images in enumerate(dataloader):
            batch_images = batch_images.cuda()
            #print('[/] batch idx: {}, forward pass...'.format(batch_idx))
            batch_embeddings = self.model(batch_images).cpu().detach().numpy()
            self.unlabelled_image_embeddings += batch_embeddings

    def assign_labels(self):
        """Assign class labels to unlabelled images."""

        if self.distance_metric == 'euclidean':
            for query_embedding in self.unlabelled_image_embeddings:
                distances = np.sum(np.square(query_embedding - self.class_reference_embeddings), axis=1)
                class_label = self.classes[np.argmin(distances)]
                self.pseudo_labels.append(class_label)

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

        # generate embeddings for reference and unlabelled images
        class_indices = self.create_reference_embeddings()
        self.create_embeddings_for_unlabelled_images()

        # assign labels based on distance metric
        self.assign_labels()

        # generate labelled data using pseudo-labels
        self.generate_labelled_data()
