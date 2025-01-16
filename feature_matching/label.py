import os
import argparse

from feature_matching_labeler import FeatureMatchingLabeler

parser = argparse.ArgumentParser()
parser.add_argument('--unlabelled-dump', help='path to unlabelled data dump.')
parser.add_argument('--reference-images', help='path to reference images which you want to retrieve.')
parser.add_argument('--result-path', help='path to save resultant labelled data.')
parser.add_argument('--viz', action="store_true", help='flag to enable visualization mode.')
parser.add_argument('--viz-path', help='path to save visualizations of detections.')

args = parser.parse_args()

if __name__ == '__main__':

    labeler = FeatureMatchingLabeler(
        args.unlabelled_dump,
        args.reference_images,
        args.result_path,
        args.viz,
        args.viz_path
    )
    labeler.label_images()