# Imports
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from io import BytesIO
from PIL import Image

def rle_encode(mask):
    """
    Convert a mask into RLE format.

    Args:
        mask (np.array): Binary mask of shape (height, width), where 1 indicates the object and 0 is the background.

    Returns:
        list: RLE encoded mask.
    """
    pixels = mask.flatten()
    # We need to allow for cases where there is a '1' at either end of the sequence.
    # We do this by padding with a zero at each end when needed.
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixels = np.concatenate([[0], pixels, [0]])
    
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2  # Convert 1-based index to 0-based
    if use_padding:
        # Removing the padding we added at the beginning
        runs = runs - 1
    
    # Every second element is the start of a new run of 1s
    runs[1::2] -= runs[::2]
    
    return runs.tolist()

# Function to convert Labelbox JSON format to Label Studio JSON format
def convert_labelbox_to_label_studio(labelbox_json_path, label_studio_json_path):

    # Load the Labelbox JSON file
    with open(labelbox_json_path, 'r') as file:
        labelbox_data = json.load(file)

    # Initialize the list for Label Studio tasks
    label_studio_tasks = []

    # Iterate over each entry in the Labelbox JSON file
    for entry in labelbox_data:
        
        # Extract image data and annotations
        image_data = entry['Labeled Data']  # Adjust this if necessary
        box_annotations = entry['Label']['objects'][0]['bbox']  # Adjust this if necessary
        line_annotations = entry['Label']['objects'][1]['line']  # Adjust this if necessary

        # Convert to Label Studio annotation format
        label_studio_annotation = {
            "result": [

                {
                'type': 'rectanglelabels',
                "from_name": "label",
                "to_name": "image",
                'original_width': 1280,  # Adjust this if necessary
                'original_height': 720,  # Adjust this if necessary
                "image_rotation": 0,
                'value': {
                    'rotation': 0, 
                    'x': box_annotations['left'] / 1280 * 100,
                    'y': box_annotations['top'] / 720 * 100,
                    'width': box_annotations['width'] / 1280 * 100,
                    'height': box_annotations['height'] / 720 * 100,
                    'rectanglelabels': ["Chicory"]
                    }
                },

                {
                'type': 'keypointlabels',
                "from_name": "keypoints",
                "to_name": "image",
                'value': {
                    'x': line_annotations[0]["x"] / 1280 * 100,
                    'y': line_annotations[0]["y"] / 720 * 100,
                    'width': box_annotations['width'] / 1280 * 100,
                    'height': box_annotations['height'] / 720 * 100,
                    'keypointlabels': ["Point1"]
                    }
                },

                {
                'type': 'keypointlabels',
                "from_name": "keypoints",
                "to_name": "image",
                'value': {
                    'x': line_annotations[1]["x"] / 1280 * 100,
                    'y': line_annotations[1]["y"] / 720 * 100,
                    'width': box_annotations['width'] / 1280 * 100,
                    'height': box_annotations['height'] / 720 * 100,
                    'keypointlabels': ["Point2"]
                    }
                },

                {
                "type":"brushlabels",
                "from_name":"segmentation",
                "to_name":"image",
                "origin":"manual",
                'original_width': 1280,  # Adjust this if necessary
                'original_height': 720,  # Adjust this if necessary
                "image_rotation": 0,
                'value': {
                    'format': "rle", 
                    "rle": [],
                    "brushlabels": ["Chicory"]
                    }
                }
                
                ]
        }

        # Create a task for Label Studio
        task = {
            'data': {
                'image': image_data
            },
            'annotations': [label_studio_annotation]
        }
        label_studio_tasks.append(task)

    # Save the converted data to a new JSON file for Label Studio
    with open(label_studio_json_path, 'w') as file:
        json.dump(label_studio_tasks, file, indent=4)

# Path to your Labelbox JSON file
labelbox_json_path = './data/export_20230705.json'

# Path where you want to save the Label Studio JSON file
label_studio_json_path = 'label_studio.json'

# Convert the file
convert_labelbox_to_label_studio(labelbox_json_path, label_studio_json_path)
