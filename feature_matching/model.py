from transformers import AutoImageProcessor, AutoModel

from config import FEATURE_MATCHING_CONFIG
from models.loftr import LoFTRModel

class Models:

    def __init__(self):
        self.batch_size = 1

    def create_superglue_arch(self, model_arch):
        processor = AutoImageProcessor.from_pretrained(model_arch)
        model = AutoModel.from_pretrained(model_arch)

        model_data = {
                "model": model,
                "processor": processor,
                "batch_size": self.batch_size
            }
        return model_data


    def create_loftr(self):

        loftr = LoFTRModel()
        processor = None
        model_data = {
                "model": loftr,
                "processor": processor,
                "batch_size": self.batch_size
            }

        return model_data


    def create_model(self):
        """Instantiate the network class and initialize with pretrained weights."""

        model_arch = FEATURE_MATCHING_CONFIG["model"]
        print(f'[/] creating {model_arch}...')

        if "superglue" in model_arch:
            return  self.create_superglue_arch(model_arch)

        elif model_arch == "loftr":
            return self.create_loftr()
