from config import OCR_CONFIG

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from doctr.models import ocr_predictor

class Models:

    def __init__(self):
        self.batch_size = 1

    def create_trocr(self, model_arch):

        processor = TrOCRProcessor.from_pretrained(model_arch)
        model = VisionEncoderDecoderModel.from_pretrained(model_arch)

        model_data = {
            "model": model,
            "processor": processor,
            "prompt": None
        }

        return model_data

    def create_doc_tr(self, td_model, tr_model):

        model = ocr_predictor(det_arch=td_model, reco_arch=tr_model, pretrained=True)
        model_data = {
            "model": model,
            "processor": None,
            "prompt": None
        }

        return model_data


    def create_model(self):

        model_arch = OCR_CONFIG["model"]
        print(f'[/] creating {model_arch}...')

        if "trocr" in model_arch:
            return self.create_trocr(model_arch)

        if model_arch == "mindee/doctr":
            return self.create_doc_tr(
                OCR_CONFIG["text_detection"],
                OCR_CONFIG["text_recognition"]
            )
