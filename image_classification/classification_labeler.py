from image2image import Image2ImageRetrieval
from text2image import Text2ImageRetrieval
from VLM import VisionLanguageModel

from config import IMAGE_CLASSIFIER_CONFIG

class ClassificationLabeler:

    def __init__(self, args):
        self.auto_label_mode = IMAGE_CLASSIFIER_CONFIG["auto_label_mode"]
        self.args = args

    def label_images(self):

        if self.auto_label_mode == "text2image":
            labeler = Text2ImageRetrieval(
                self.args.class2prompts,
                self.args.unlabelled_dump,
                self.args.result_path
            )
            labeler.label_images()