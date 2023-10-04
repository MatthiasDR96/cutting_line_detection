# Cutting line detection of chicory

## Folder structure

## Content

* config: contains the parameter files that are used in the scripts. 
* data: contains the Labelbox render and is the default location to store the images and labels as well as the generated Pandas DataFrames
* models: contains all trained models
* notebooks: contains Jupyter Notebooks to transfer the knowledge to a broad audience
* results: contains all training results in csv files and images
* scripts contains all scripts for executing the software as well as a utils.py file that contains all neccesary functions. 
* tests: contains all scripts for testing the functionalities in this package

### Data generation

In the data generation script, the software loops over all 216 entries in the .json render from LabelBox. For every entry it downloads the image, retrieves the bounding box and converts it to format [xmin, ymin, xmax, ymax], and retrieves the line and converts it to format [x1, y1, x2, y2]. It saves the images with their respective ID in the /data/images folder, and the labels as .txt files in the /data/labels folder. The total format of the label is [xmin, ymin, xmax, ymax, x1, y1, x2, y2].

### Data analysis

In the data analysis script, the software loops over all images in the /data/images folder and retrieves the corresponding label. To visualize the correctness of the data, it plots the image with the corresponding annotated bounding box (red) and cutting line (green). Also the cutting line in carthesian coordinates is drawn in violet. 

### Data preparation

In the data preparation step, all data is split into training, validation, and test data so that we have 70% training data (151 images), 15% validation data (32 images) and 15% test data (33 images). The image paths are stored in three Pandas DataFrames train_data.csv, valid_data.csv, and test_data.csv. These are later used to feed to the neural network. 

### Model training  

For model training we use a pretrained ResNet18 model architecture where we keep the feature layers but remove the classification layers. We add a new and untrained classification layer with four output neurons for predicting two points [x1, y1, x2, y2]. We train for 100 epochs with a batch size of 16, and a learning rate of 0.01. 

Before being fed to the neural network, several augmentations are performed on the images and labels using the Albumentations package. Especially resizing to 224x224, normalizing and converting to a tensor are applied. The labels are also transformed using the same transformations. As a loss function, a combination of two losses is used: the error between the first predicted point and the middle point of the line lable, and the angle error between the true and predicted line using cosine similarity. 

### Model validation

To validate the correctness of the model visually, the software loops over all images in the test dataset and draws the ground truth line and the predicted line. Also the total loss is computed. 

### Notebooks

All notebooks can be accessed via: https://matthiasdr96.github.io/cutting_line_detection/intro.html




