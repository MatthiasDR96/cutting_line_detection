# Cutting line detection of chicory

In this set of notebooks, the optimal cutting line to cut the loaf from the root of chicory is determined using a neural network. A labeled dataset of 216 images of uncut chicory is available. The cutting lines are labeled using two points and the label has a format [x1, y1, x2, y2]. A pretrained Resnet18 model is used to predict the optimal cutting line. As a cost function, the Euclidean distance from the first predicted point to the center of the two labeled points is summed with the angle between the two lines in degrees. 

The first notebook 'Data analysis' loops over all images and draws the bounding box and cutting line labels. The second notebook 'Data preparation' splits the dataset in a train, validation, and test set. The third script 'Model training' trains a neural network on the training set and outputs the learning curves as well as the model. The last script 'Model validation' shows some predictions on the testset. 

```{tableofcontents}
```
