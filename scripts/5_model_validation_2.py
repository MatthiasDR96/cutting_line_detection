# Imports
import torch
import numpy as np
import pandas as pd
from utils import *
import matplotlib.pyplot as plt

# Model file name
model_file_name = 'model_0.001-16-0-100'

# Check if there is a graphical card (CUDA) available on the PC
device = torch.device("cuda:" + str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")

# Load datasets
df_test = pd.read_pickle('./data/train_data.csv')

# Load model
model = torch.load("./models/" + model_file_name + ".pt").to(device)

# Set model to evaluate mode     
model.eval()  

# Convert all images and labels
loss_cos = 0
loss_rmse = 0
loss = 0
for i, row in df_test.iterrows():

	# Define subplot
	plt.subplot(3, 4, i+1)

	# Get the image path name
	img_path = row["path_names"]

	# Read image
	image = cv2.imread(img_path) 
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	height, width, channels = image.shape

	# Get label
	key = img_path.split('\\')[-1].split('.')[0]
	label_file = open("./data/labels/" + key + ".txt", "r")
	labeldata = label_file.read().split(" ")

	# Get keypoints
	y = [(float(labeldata[4]), float(labeldata[5])), (float(labeldata[6]), float(labeldata[7]))]

	# Define augmentation pipeline
	transform = A.Compose([
		A.Resize(224, 224),
		A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		ToTensorV2(),
	], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))

	# Transform
	transformed = transform(image=image, keypoints=y)
	transformed_image = transformed['image']

	# Get ground truth line
	a1 = 0 if (y[1][0] - y[0][0]) == 0 else (y[1][1] - y[0][1]) / (y[1][0] - y[0][0])
	b1 = y[0][1] - a1*y[0][0]

	# Predict line 
	pred = model(transformed_image.unsqueeze(0).cuda().float()) # Unsqueeze to solve for missing batch dimension
	pred = pred.detach().cpu().numpy()[0]

	# Scale prediction to account for resizing
	pred[0] = pred[0] /224*width
	pred[1] = pred[1] /224*height
	pred[2] = pred[2] /224*width
	pred[3] = pred[3] /224*height

	# Get predicted line
	a2 = 0 if (pred[2] - pred[0]) == 0 else (pred[3] - pred[1]) / (pred[2] - pred[0])
	b2 = pred[1] - a2*pred[0]

	# Draw ground truth and predicted line
	draw_extended_line(image, a1, b1, color=(0, 255, 0), thickness=3)
	draw_extended_line(image, a2, b2, color=(255, 0, 0), thickness=3)

	# Visualize
	cv2.rectangle(image, (int(labeldata[0]), int(labeldata[1])), (int(labeldata[2]), int(labeldata[3])), (255, 0, 0), 1)
	plt.imshow(image)

	# Get x and y differences between predicted points
	delta_x_pred = pred[2]-pred[0]
	delta_y_pred = pred[3]-pred[1]

	# Get x and y differences between ground truth points
	delta_x_y = y[1][0]-y[0][0]
	delta_y_y = y[1][1]-y[0][1]

	# Get center of lable
	y_centerx = (y[1][0]+y[0][0])/2
	y_centery = (y[1][1]+y[0][1])/2

	# Get vectors of deltas
	a = [delta_x_pred, delta_y_pred]
	b = [delta_x_y, delta_y_y]

	# Compute the sum of angle differences in degrees
	loss_cos_value = np.arccos(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))* 180 / math.pi
	loss_rmse_value = math.sqrt((pred[0] - y_centerx)**2 + (pred[1] - y_centery)**2)

	# Add info to image
	#plt.title('Angle error - ' + str(round(loss_cos_value,2)))

	# Add losses
	loss_cos += loss_cos_value
	loss_rmse += loss_rmse_value
	loss += loss_cos_value + loss_rmse_value

	# Stop reading
	if i >= 3*4-1: break

# Show
plt.show()

#fig.savefig('./results/train_results/' + str(i) + '.jpg')

# Print results
print("The total mean distance between the points is " + str(loss_rmse/len(df_test)))
print("The total mean angular distance between the lines " + str(loss_cos/len(df_test)))
print("The total mean loss is " + str(loss/len(df_test)))

	

	