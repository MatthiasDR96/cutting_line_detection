### This script splits the dataset in train, validation, and test sets and saves the image file names as csv files for each set.

# Imports
import glob
import random
import pandas as pd
from utils import *

# Folder to save data
root_folder = './data'

# Init lists
image_paths = glob.glob(root_folder + '/images/*jpg') 

# Shuffle image paths
random.shuffle(image_paths)

# Split train an val (70/30)
train_image_paths = image_paths[:int(0.7*len(image_paths))]
valid_image_paths = image_paths[int(0.7*len(image_paths)):] 

# Split valid futher in val and test (50/50)
valid_image_paths_ = valid_image_paths[:int(0.5*len(valid_image_paths))]
test_image_paths = valid_image_paths[int(0.5*len(valid_image_paths)):]

# Save to Dataframes
df_train = pd.DataFrame(train_image_paths, columns=['path_names'])
df_valid = pd.DataFrame(valid_image_paths_, columns=['path_names'])
df_test = pd.DataFrame(test_image_paths, columns=['path_names'])

# Save DataFrame
df_train.to_pickle(root_folder + '/train_data.csv')
df_valid.to_pickle(root_folder + '/valid_data.csv')
df_test.to_pickle(root_folder + '/test_data.csv')

# Print result
print("Train size: {}\nValid size: {}\nTest size: {}".format(len(train_image_paths), len(valid_image_paths_), len(test_image_paths)))