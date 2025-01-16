import os
import cv2
import json
import numpy as np

from PIL import Image
from tqdm import tqdm

import torch
from transformers.image_utils import load_image

from config import VQA_CONFIG
from model import Models

class VisionLanguageModel:
    def __init__(self, unlabelled_dump_path, result_path):

        self.unlabelled_dump_path = unlabelled_dump_path
        self.unlabelled_image_paths, self.pseudo_labels = [], {}
        self.result_path = result_path

        self.model_data = Models().create_model()

    def label_images_using_vlm(self):
        """Label unlabelled data using Vision Language Models."""

        files = os.listdir(self.unlabelled_dump_path)
        for file in files:
            self.unlabelled_image_paths.append(os.path.join(self.unlabelled_dump_path, file))

        model, processor, prompt = self.model_data["model"], self.model_data["processor"], self.model_data["prompt"]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        max_new_tokens = 100

        if "qwen" in VQA_CONFIG["model"]:
            for image_path in self.unlabelled_image_paths:
                image = Image.open(image_path)
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image,
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

                # Preparation for inference
                # text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(
                    text=[prompt], images=[image], padding=True, return_tensors="pt"
                )
                inputs = inputs.to("cuda")

                # Inference: Generation of the output
                generated_ids = model.generate(**inputs, max_new_tokens=128)
                generated_ids = [
                    output_ids[len(input_ids):]
                        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
                    ]
                output_text = processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                print(output_text)

        else:
            for image_path in self.unlabelled_image_paths:
                if "paligemma" in VQA_CONFIG["model"]:
                    image = load_image(image_path)
                else:
                    image = Image.open(image_path)

                inputs = processor(text=prompt, images=image, return_tensors="pt").to(torch.float16).to(device)
                generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
                generated_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

                self.pseudo_labels[image_path] = generated_text

        with open(self.result_path, "w") as f:
            json.dump(self.pseudo_labels, f, indent=2)


    def label_images(self):
        """Labels image classification data based on the inputs config params."""

        self.label_images_using_vlm()