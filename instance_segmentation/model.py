from transformers import pipeline
# from segment_anything import sam_model_registry, SamPredictor

from config import INSTANCE_SEGMENTOR_CONFIG

class Models:

    def __init__(self):
        self.batch_size = 1
        self.device = "cuda"

    def create_model(self):
        """Instantiate the network class and initialize with pretrained weights."""

        print('[/] creating model...')
        model_arch = INSTANCE_SEGMENTOR_CONFIG["model"]
        label_mode = INSTANCE_SEGMENTOR_CONFIG["auto_label_mode"]

        if model_arch == "SAM-ViT":
            if label_mode == "zero-shot":
                generator = pipeline("mask-generation", model="facebook/sam-vit-base", device=self.device)
                # sam_checkpoint = "/tmp/models/sam_vit_h_4b8939.pth"
                # model_type = "vit_h"

                # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                model_data = {
                    "model": generator,
                    "processor": None,
                    "tokenizer": None,
                    "batch_size": self.batch_size
                }

                return model_data