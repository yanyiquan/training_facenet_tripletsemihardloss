from crop_image import cropping_image
import numpy as np
import tensorflow as tf
import keras
import os
import cv2

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path")

args = parser.parse_args()

cropping_image(args.path)