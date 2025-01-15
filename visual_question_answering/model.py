from config import VQA_CONFIG

import torch
from transformers import(
    AutoProcessor, AutoModelForImageTextToText, AutoModelForVision2Seq,
    PaliGemmaProcessor, PaliGemmaForConditionalGeneration,
    LlavaNextProcessor, LlavaNextForConditionalGeneration
)
import open_clip

class Models:

    def __init__(self):
        self.batch_size = 1
        self.prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"

    def create_llava_next_arch(self):

        processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        model = AutoModelForImageTextToText.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
            )

        model_data = {
            "model": model,
            "processor": processor,
            "prompt": self.prompt
        }

        return model_data

    def create_smol_vlm(self):

        processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
        model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceTB/SmolVLM-Instruct",
            torch_dtype=torch.float16,
            _attn_implementation="eager"
        )

        model_data = {
            "model": model,
            "processor": processor,
            "prompt": self.prompt
        }

        return model_data

    def create_pali_gemma(self, model_arch):

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_arch,
            torch_dtype=torch.float16,
            ).eval()
        processor = PaliGemmaProcessor.from_pretrained(model_arch)

        prompt = ""
        model_data = {
            "model": model,
            "processor": processor,
            "prompt": prompt
        }

        return model_data

    def create_model(self):

        model_arch = VQA_CONFIG["model"]
        print(f'[/] creating {model_arch}...')

        if model_arch == "LLaVA-NeXT":
            return self.create_llava_next_arch()

        elif model_arch == "HuggingFaceTB/SmolVLM-Instruct":
            return self.create_smol_vlm()

        elif "paligemma" in model_arch:
            return self.create_pali_gemma(model_arch)