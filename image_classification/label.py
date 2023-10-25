import os
import argparse

from classification_labeler import ClassificationLabeler

parser = argparse.ArgumentParser()
parser.add_argument('--unlabelled-dump', help='path to unlabelled data dump')
parser.add_argument('--class2prompts', help='path to json file containing class name and corresponding prompt')
parser.add_argument('--result-path', help='path to save resultant labelled data')

args = parser.parse_args()

if __name__ == '__main__':

    labeler = ClassificationLabeler(args)
    labeler.label_images()