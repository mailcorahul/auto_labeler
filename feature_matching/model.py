from transformers import AutoImageProcessor, AutoModel

from config import FEATURE_MATCHING_CONFIG

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

    def create_model(self):
        """Instantiate the network class and initialize with pretrained weights."""

        print('[/] creating model...')
        model_arch = FEATURE_MATCHING_CONFIG["model"]

        if "superglue" in model_arch:
            model_data = self.create_superglue_arch(model_arch)

        return model_data

