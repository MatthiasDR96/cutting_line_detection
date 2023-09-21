# Imports
import cv2
import math
import torch
import numpy as np
import albumentations as A
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

# This function implements the training and validation of the neural network
def train_model(model, optimizer, dataloaders, device, epochs):

	# Learning curves
	results = {"training_loss": [], "training_loss_rmse": [] , "training_loss_cos": [], "validation_loss": [], "validation_loss_rmse": [] , "validation_loss_cos": []}

	# Loop over epochs
	best_model = model
	best_loss = float('inf')
	best_loss_rmse = 0.0
	best_loss_cos = 0.0
	for epoch in range(epochs):

		# Debug
		print(f'Epoch {epoch+1}/{epochs}')

		# Do a train and validation phase for each epoch
		for phase in ['train', 'valid']:

			# Set state of model corresponding to phase
			if phase == 'train':
				model.train()  # Set model to training mode
			else:
				model.eval()   # Set model to evaluate mode

			# Reset loss over all batches and number of samples 
			running_loss_rmse = 0.0
			running_loss_cos = 0.0
			running_loss = 0.0
			num_samples = 0

			# Iterate over batches of inputs
			for x, y in dataloaders[phase]:

				# Inputs shape
				batch_size, *_ = x.shape

				# Convert input and outputs to CUDA
				x = x.to(device)
				y = y.to(device)

				# Predict line
				pred = model(x)

				# Compute total loss over batch
				loss_rmse_values = loss_rmse_fun(pred, y)
				loss_cos_values = loss_cos_fun(pred, y)
				loss = loss_rmse_values + loss_cos_values

				# Debug
				#debug(x, pred, y)

				# Sum losses of all batches
				running_loss_rmse += loss_rmse_values.item()
				running_loss_cos += loss_cos_values.item()
				running_loss += loss.item()

				# Calculate total samples
				num_samples += batch_size

				# When training
				if phase == 'train':
					optimizer.zero_grad() # Clear out gradients
					loss.backward() # Do backpropagation
					optimizer.step() # Step the optimizer

			# Compute total loss over the whole dataset (sum of loss of all samples divided by total number of samples)
			dataset_loss_rmse = running_loss_rmse/num_samples
			dataset_loss_cos = running_loss_cos/num_samples
			dataset_loss = running_loss/num_samples

			# Save results
			if phase == 'train':
				results['training_loss'].append(dataset_loss)
				results['training_loss_rmse'].append(dataset_loss_rmse)
				results['training_loss_cos'].append(dataset_loss_cos)
				print("train_loss %.3f - %.3f" % (dataset_loss_rmse, dataset_loss_cos))
			elif phase == 'valid':
				results['validation_loss'].append(dataset_loss)
				results['validation_loss_rmse'].append(dataset_loss_rmse)
				results['validation_loss_cos'].append(dataset_loss_cos)
				print("val_loss %.3f - %.3f" % (dataset_loss_rmse, dataset_loss_cos))

			# Save model
			if phase == 'valid' and dataset_loss < best_loss:
				best_model = model
				best_loss = dataset_loss
				best_loss_rmse = dataset_loss_rmse
				best_loss_cos = dataset_loss_cos

		# Debug
		print('-' * 10)

	return best_loss, best_loss_rmse, best_loss_cos, best_model, results


# Shows batches
def debug(x, pred, y):

	# Extract tensors
	x_tmp = x.detach().cpu().numpy()
	pred_tmp = pred.detach().cpu().numpy()
	y_tmp = y.detach().cpu().numpy()

	# Get subplots size
	size = int(math.sqrt(len(x_tmp)))

	# Plot images
	plt.subplots(size, size)
	for i in range(len(x_tmp)):

		# Define subplot
		plt.subplot(size, size, i+1)

		# Get image info
		image = x_tmp[i]
		prediction = pred_tmp[i]
		label = y_tmp[i]
		
		# Compute label center
		y_centerx = (label[0] + label[2])/2
		y_centery = (label[1] + label[3])/2

		# Plot points
		plt.plot([prediction[0]], [prediction[1]], 'b*')
		plt.plot([prediction[2]], [prediction[3]], 'r*')
		plt.plot(y_centerx, y_centery, 'g*')

		# Decode lines
		a1, b1 = line_decode(prediction, 3)
		a2, b2 = line_decode(label, 3)

		# Plot lines
		image = image.transpose((1, 2, 0)).copy() 
		im = draw_extended_line(image, a1, b1, color=(255, 0, 0), thickness=1)
		im = draw_extended_line(im, a2, b2, color=(0, 255, 0), thickness=1)

		# Show image
		plt.imshow(im)

	# Show plots
	plt.show()
	plt.draw()
	plt.pause(0.01)


# Computes the error angle between the predicted and ground truth lines in degrees
def loss_cos_fun(pred, y):

	# Get x and y differences between predicted points
	delta_x_pred = pred[:,2]-pred[:,0]
	delta_y_pred = pred[:,3]-pred[:,1]

	# Get x and y differences between ground truth points
	delta_x_y = y[:,2]-y[:,0]
	delta_y_y = y[:,3]-y[:,1]

	# Stack both into new tensors
	a = torch.stack((delta_x_pred, delta_y_pred), 1)
	b = torch.stack((delta_x_y, delta_y_y), 1)

	# Compute the sum of anle differences in degrees
	loss = torch.sum(torch.acos(F.cosine_similarity(a, b, dim=1)) * 180 / math.pi)

	return loss


# Computes the Euclidean distance between the first predicted point and the center of the ground truth points
def loss_rmse_fun(pred, y):

	# Only get first xy point of prediction
	xy1_pred = torch.stack((pred[:,0], pred[:,1]), 1) 

	# Get center of two ground truth points
	y_centerx = (y[:,0] + y[:,2])/2
	y_centery = (y[:,1] + y[:,3])/2
	y_center = torch.stack((y_centerx, y_centery), 1)

	# Compute sum of Euclidean distances between first predicted point and center point
	loss = torch.sum(torch.sqrt(torch.sum((xy1_pred - y_center)**2, 1)))

	return loss


# This function decodes polar coordinates rho*cos(theta) and rho*sin(theta) into a slope and an intercept
def line_decode(target):

	# Get target
	x1 = target[0]
	y1 = target[1]
	x2 = target[2]
	y2 = target[3]

	# Convert line to Cartesian coordinates
	a = 0 if (x2 - x1) == 0 else (y2 - y1) / (x2 - x1)
	b = y1 - a*x1

	return a, b


# This function draws an extended line on an image based on a slope and an intercept
def draw_extended_line(image, a, b, color=(255, 0, 0), thickness=1):
		
	# Get image shape
	_, cols,*_ = image.shape

	# Create line points
	start_point = (0, int(a*0 + b))
	end_point = (int(cols), int(a*cols + b))
	
	# Draw the extended line in red
	image = cv2.line(image, start_point, end_point, color, thickness)
	
	return image


# This function draws a polar line on an image based on a rho and a theta
def draw_polar_line(image, R, theta):
	
	# Compute intersection point
	Sy = R * math.sin(theta)
	Sx = R * math.cos(theta)
	
	# Compute line points
	start_point = (0, 0)
	end_point = (int(Sx), int(Sy))

	# draw the perpendicular line in blue
	image = cv2.line(image, start_point, end_point, color=(0,0,255), thickness=2)
		
	return image


# This class impements the Dataset
class Dataset(Dataset):

	def __init__(self, paths, transform=False):
		self.paths = paths
		self.transform = transform

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, idx):

		# Retrieve data sample
		img_path = self.paths[idx]

		# Read image and convert to RGB space
		image = cv2.imread(img_path) 
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# Get label
		key = img_path.split('\\')[-1].split('.')[0]
		label_file = open("./data/labels/" + key + ".txt", "r")
		labeldata = label_file.read().split(" ")

		# Get keypoints
		keypoints = [(float(labeldata[4]), float(labeldata[5])), (float(labeldata[6]), float(labeldata[7]))]

		# Define augmentation pipeline
		if self.transform:
			transform = A.Compose([
				A.AdvancedBlur(),
				A.RandomGravel(),
				A.ISONoise(),
				A.RandomRotate90(p=0.5),
				A.HorizontalFlip(p=0.2),
				A.RandomBrightnessContrast(p=0.5),
				A.Resize(224, 224),
				A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
				ToTensorV2(),
			], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))
		else:
			transform = A.Compose([
				A.Resize(224, 224),
				A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
				ToTensorV2(),
			], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))
		
		# Transform image and keypoints
		transformed = transform(image=image, keypoints=keypoints)
		transformed_image = transformed['image']
		transformed_keypoints = transformed['keypoints']

		# Get target
		target = np.array([transformed_keypoints[0][0], transformed_keypoints[0][1], transformed_keypoints[1][0], transformed_keypoints[1][1]])

		return transformed_image, target
