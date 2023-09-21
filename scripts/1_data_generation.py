### This script converts the Labelbox render (.json) to yolo/resnet format (.png en .txt)

# Imports
import os
import numpy
import requests
import pandas as pd
from PIL import Image
from io import BytesIO

# Render from Labelbox
render = './data/export_20230705.json'

# Folder to save data
savefolder = './data'

# Open Labelbox render
df = pd.read_json(render)

# Make folders
try:
    os.mkdir(savefolder + "/labels")
    os.mkdir(savefolder + "/images")
except:
    pass

# Convert all images and labels
for i, row in df.iterrows():

    # Print update
    print(i+1,"/", len(df))

    # Get the ID of the sample
    sample_id = row["ID"]

    # If the sample has a label, extract image and label
    if row["Label"]["objects"] != []:

        # Get image of sample
        url = row["Labeled Data"]
        while True:
            try:
                response = requests.get(url)
                break
            except:
                response = 0

        # Get image
        img = Image.fromarray(numpy.array(Image.open(BytesIO(response.content)).convert('RGB')))
        img.save("./data/images/" + sample_id + ".jpg")

        # Get bounding box 
        bbox = row["Label"]["objects"][0]["bbox"] # In format [top, left, height, width]
        xmin = bbox["left"]
        ymin = bbox["top"]
        xmax = bbox["left"] + bbox["width"]
        ymax = bbox["top"] + bbox["height"]
        
        # Get line
        line = row["Label"]["objects"][1]["line"]
        x1 = line[0]["x"]
        y1 = line[0]["y"]
        x2 = line[1]["x"]
        y2 = line[1]["y"]

    # Write label
    with open(savefolder + "/labels/"+ sample_id + ".txt", 'w') as f:
        f.write(str(xmin) + " " + str(ymin) + " " + str(xmax) + " " + str(ymax) + " " + str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2))
