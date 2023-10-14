from transformers import Owlv2Processor, Owlv2ForObjectDetection

from config import OBJECT_DETECTOR_CONFIG

class Models:

    def __init__(self):
        self.batch_size = 1

    def create_model(self):
        """Instantiate the network class and initialize with pretrained weights."""

        print('[/] creating model...')
        model_arch = OBJECT_DETECTOR_CONFIG["model"]

        if model_arch == "OWL-ViT-v2":
            processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
            model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

            model_data = {
                "model": model,
                "processor": processor,
                "tokenizer": None,
                "batch_size": self.batch_size
            }

            return model_data