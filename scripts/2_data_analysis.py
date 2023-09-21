### This script loops over the images in the dataset and visualizes the boundingbox label in red and the cutting line in violet.

# Imports
import glob
import cv2
import numpy as np
from utils import *
import matplotlib.pyplot as plt

# Get all image path names
img_paths = glob.glob('./data/images/*.jpg')

# Loop over all images
for img_path in img_paths:

    # Read image
    im = cv2.imread(img_path) 
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    height, width, depth = im.shape

    # Get label
    key = img_path.split('\\')[-1].split('.')[0]
    label_file = open("./data/labels/" + key + ".txt", "r")
    labeldata = label_file.read().split(" ")
    line_label = np.array([float(labeldata[4]), float(labeldata[5]), float(labeldata[6]), float(labeldata[7])])

    # Convert line to polar coordinates
    a = 0 if (line_label[2] - line_label[0]) == 0 else (line_label[3] - line_label[1]) / (line_label[2] - line_label[0])
    b = line_label[1] - a*line_label[0]

    # Draw extended line
    im = draw_extended_line(im, a, b, color=(255, 0, 255), thickness=4)

    # Draw ground truth rectangle and line (based on two points)
    cv2.line(im, (int(line_label[0]), int(line_label[1])), (int(line_label[2]), int(line_label[3])), (0, 255, 0), 3) 
    cv2.rectangle(im, (int(labeldata[0]), int(labeldata[1])), (int(labeldata[2]), int(labeldata[3])), (255, 0, 0), 3)
    plt.imshow(im)
    plt.draw()
    plt.pause(0.001)