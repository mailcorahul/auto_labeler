from VLM import VisionLanguageModel

from config import VQA_CONFIG

class VQALabeler:

    def __init__(self, args):
        self.auto_label_mode = VQA_CONFIG["auto_label_mode"]
        self.args = args

    def label_images(self):

        if self.auto_label_mode == "VLM":
            labeler = VisionLanguageModel(
                self.args.unlabelled_dump,
                self.args.result_path
            )
            labeler.label_images()