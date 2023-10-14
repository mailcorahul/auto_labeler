import os
import argparse

from detection_labeler import DetectionLabeler

parser = argparse.ArgumentParser()
parser.add_argument('--unlabelled-dump', help='path to unlabelled data dump.')
parser.add_argument('--class-texts-path', help='path to json containing list of texts descriptions of objects to detect in a zero shot setting.')
parser.add_argument('--prompt-images', help='path to prompt images for guided one shot detection.')
parser.add_argument('--result-path', help='path to save resultant labelled data.')
parser.add_argument('--viz', action="store_true", help='flag to enable visualization mode.')
parser.add_argument('--viz-path', help='path to save visualizations of detections.')

args = parser.parse_args()

if __name__ == '__main__':

    labeler = DetectionLabeler(
        args.unlabelled_dump,
        args.class_texts_path,
        args.result_path,
        args.prompt_images,
        args.viz,
        args.viz_path
    )
    labeler.label_images()