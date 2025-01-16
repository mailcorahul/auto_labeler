from ocr import OCR

from config import OCR_CONFIG

class OCRLabeler:

    def __init__(self, args):
        self.auto_label_mode = OCR_CONFIG["auto_label_mode"]
        self.args = args

    def label_images(self):

        if self.auto_label_mode == "OCR":
            labeler = OCR(
                self.args.unlabelled_dump,
                self.args.result_path
            )
            labeler.label_images()