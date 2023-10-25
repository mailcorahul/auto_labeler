# auto_labeler - A library to automatically label any computer vision dataset at zero/near-zero manual labeling cost.

# Introduction

**auto_labeler** is a simple, easy-to-use framework which helps you generate high quality pseudo-labels for any given computer vision task at hand. This library abstracts away all the various SOTA algorithms in play for Computer Vision, their modes of usage(like** image based retrieval, text to image retrieval, feature matching for classification, zero shot or single shot promptable detection or instance segmentation or exploiting LLMs**) to auto label any vision dataset.

# Features

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

To be released in the near future:
1. **Feature/Keypoint Matching** for Image Retrieval applications
2. **Vision Language Models(VLMs)** for image labeling tasks.

# Usage

**Setup repo and environment:**

```
git clone git@github.com:mailcorahul/auto_labeler.git
cd auto_labeler/
```

**Setup your python virtualenv(preferably python3.8):**
```
pip install -r requirements.txt
```

**To label any dataset for image classification task:**
```
cd image_classification
python label.py --unlabelled-dump 'path to the unlabelled dataset containing images'
 --class2prompts 'path to a json containing class names along with its text prompts if already known(optional)'
 --result-path 'root folder path to save the auto labeled classification data'
```

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

**To label any dataset for instance segmentation task:**
```
cd instance_segmentation
python label.py --unlabelled-dump 'path to the unlabelled dataset containing images'
 --class-texts-path 'path to a json containing the list of class objects to segment'
 --result-path 'path to .pkl file to save the auto labeled segmentation data'
 --viz 'False(set to True if visualization is required)'
 --viz-path 'path to save mask visualizations'
```
