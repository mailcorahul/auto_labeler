# auto_labeler
A library to automatically label any computer vision dataset at zero/near-zero manual labeling cost.

## Table of Contents:

- [Introduction](#introduction)
- [Installation](#installation)
- [Getting Started](#getting-started)
    - [Model Zoo](#model-zoo)
    - [Image Classification](#image-classification)
    - [Object Detection](#object-detection)
    - [Instance Segmentation](#instance-segmentation)
    - [Visual Question Answering](#visual-question-answering)
    - [Feature Matching](#feature-matching)
    - [OCR](#ocr)


## Introduction

**auto_labeler** is a simple, easy-to-use framework which helps you generate high quality pseudo-labels for any given computer vision task at hand. This library abstracts away all the various SOTA algorithms in play for Computer Vision, their modes of usage(like** image based retrieval, text to image retrieval, feature matching for classification, zero shot or single shot promptable detection or instance segmentation or exploiting LLMs**) to auto label any vision dataset.

### Features:

**1. High Abstraction**:
auto_labeler provides inbuilt wrappers over exisiting widely used frameworks such as HuggingFace, facebook-research and other key repos which are the sources of key SOTA architecures and models. Its generic, uniform interface provides easy access to most of the SOTA Computer Vision techniques, abstracting away most of the information which can be otherwise ignored by researchers, companies.

**2. Modular Interface**:
auto_labeler aims at building independent, specific modules for each vision task, making it easy to add new algorithms or modify the existing ones as needed.

**3. Minimal Touchpoints for Faster Labeling**:
auto_labeler requires minimal work from the user before labeling process can start. A few configuration changes to choose the architecture, weights to use(along with some hyper-param changes) and a simple to use **label.py** generic inference wrapper takes care of the rest.

**4. Support for a multitude of Vision Tasks:**
The library supports several vision tasks currently which includes(architectures supported are CLIP, OWL-ViT-V2, SAM-ViT)
1. **Image Classification** - with image to image retrieval and text based retrieval
2. **Object Detection** - under zero shot text based and promptable image based settings
3. **Instance Segmentation** - under zero shot setting
4. **Feature/Keypoint Matching** for Image/Instance Retrieval and 2D Image Correspondence applications
5. **Visual Question Answering(VQA)** - with the help of vision language models
6. **Optical Character Recognition(OCR)** - using Text Detection + Recognition as well as end-to-end learnable architectures.

## Installation

1. Setup repository

```
git clone git@github.com:mailcorahul/auto_labeler.git
cd auto_labeler/
```

2. Create virtualenv python enviroment(preferably above python3.8)
```
virtualenv -p python3.8 "path to autolabeler environment"
pip install -r requirements.txt
```

## Getting Started

### Model Zoo

List of architectures supported for various vision tasks

#### 1. Image Classification:
- [CLIP](https://github.com/mlfoundations/open_clip)

#### 2. Object Detection:
- [OWL-ViT-v2](https://huggingface.co/docs/transformers/model_doc/owlvit)

#### 3. Instance Segmentation:
- [Segment Anything(SAM)](https://huggingface.co/docs/transformers/main/model_doc/sam)


### Image Classification

**To label any dataset for image classification task:**
```
cd image_classification
python label.py --unlabelled-dump 'path to the unlabelled dataset containing images'
 --class2prompts 'path to a json containing class names along with its text prompts if already known(optional)'
 --result-path 'root folder path to save the auto labeled classification data'
```

### Object Detection

**To label any dataset for object detection task:**
```
cd object_detection
python label.py --unlabelled-dump 'path to the unlabelled dataset containing images'
 --class-texts-path 'path to a json containing the list of class objects to detect'
 --prompt-images 'path to prompt images for guided one shot detection'
 --result-path 'path to .json file to save the auto labeled detection data'
 --viz 'False(set to True if visualization is required)'
 --viz-path 'path to save detection bbox visualizations'
```

### Instance Segmentation

**To label any dataset for instance segmentation task:**
```
cd instance_segmentation
python label.py --unlabelled-dump 'path to the unlabelled dataset containing images'
 --class-texts-path 'path to a json containing the list of class objects to segment'
 --result-path 'path to .pkl file to save the auto labeled segmentation data'
 --viz 'False(set to True if visualization is required)'
 --viz-path 'path to save mask visualizations'
```

### Visual Question Answering


### Feature Matching


### OCR



