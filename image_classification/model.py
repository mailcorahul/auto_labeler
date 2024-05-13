from config import IMAGE_CLASSIFIER_CONFIG

from transformers import AutoProcessor, LlavaNextForConditionalGeneration
import open_clip

class Models:

    def __init__(self):
        self.batch_size = 128

    def create_clip_arch(self):
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained='laion2b_s34b_b79k'
        )
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        model_data = {
            "model": model,
            "preprocess": preprocess,
            "tokenizer": tokenizer,
            "batch_size": self.batch_size
        }

        print('[/] CLIP instantiated!')

        return model_data

    def create_llava_next_arch(self):

        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf")
        processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"

        model_data = {
            "model": model,
            "processor": processor,
            "prompt": prompt
        }

        return model_data

    def create_model(self):
        """Instantiate the network class and initialize with pretrained weights."""

        print('[/] creating model...')
        model_arch = IMAGE_CLASSIFIER_CONFIG["model"]
        if model_arch == "clip":
            return self.create_clip_arch()

        elif model_arch == "blip":
            pass

        elif model_arch == "blip2":
            pass

        elif model_arch == "LLaVA-NeXT":
            return self.create_llava_next_arch()