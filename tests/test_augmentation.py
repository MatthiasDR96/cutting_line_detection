# Imports
import cv2
import glob
import numpy
import albumentations as A
import matplotlib.pyplot as plt
from torchvision import transforms
from albumentations.pytorch import ToTensorV2

# Define augmentation pipeline
transform = A.Compose([
        A.AdvancedBlur(),
        A.RandomGravel(),
        A.ISONoise(),
        A.RandomRotate90(),
        A.HorizontalFlip(),
        A.RandomBrightnessContrast(),
        A.Resize(224, 224),
        #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))

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

    # Get keypoints
    keypoints = [(float(labeldata[4]), float(labeldata[5])), (float(labeldata[6]), float(labeldata[7]))]

    # Draw line
    im_tmp = cv2.line(im.copy(), (int(keypoints[0][0]), int(keypoints[0][1])), (int(keypoints[1][0]), int(keypoints[1][1])), (0, 255, 0), 3) 

    # Transform
    transformed = transform(image=im, keypoints=keypoints)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['keypoints']

    # Draw line
    transformed_image = transformed_image.numpy().transpose((1, 2, 0)).copy()
    cv2.line(transformed_image, (int(transformed_bboxes[0][0]), int(transformed_bboxes[0][1])), (int(transformed_bboxes[1][0]), int(transformed_bboxes[1][1])), (0, 255, 0), 3) 
    
    # Show results
    plt.subplot(1, 2, 1)
    plt.imshow(im_tmp)
    plt.subplot(1, 2, 2)
    plt.imshow(transformed_image)
    plt.show()