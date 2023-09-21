# Imports
import torch
from utils import *
import pandas as pd
from torch.utils.data import DataLoader

# Model file name
model_file_name = 'model_0.01-16-0-100'

# Check if there is a graphical card (CUDA) available on the PC
device = torch.device("cuda:" + str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")

# Load model
model = torch.load("./models/" + model_file_name + ".pt").to(device)

# Set model to evaluate mode     
model.eval()  

# Load dataset
df = pd.read_pickle('./data/test_data.csv')

# Create dataset
ds = Dataset(df['path_names'], transform=False)

# Create data loader
dataloader = DataLoader(ds, batch_size=10, shuffle=True)

# Reset loss over all batches and number of samples 
running_loss_rmse = 0.0
running_loss_cos = 0.0
running_loss = 0.0
num_samples = 0

# Iterate over batches of inputs
for x, y in dataloader:

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

	# Sum losses of all batches
	running_loss_rmse += loss_rmse_values.item()
	running_loss_cos += loss_cos_values.item()
	running_loss += loss.item()

	# Calculate total samples
	num_samples += batch_size

# Compute total loss over the whole dataset (sum of loss of all samples divided by total number of samples)
dataset_loss_rmse = running_loss_rmse/num_samples
dataset_loss_cos = running_loss_cos/num_samples
dataset_loss = running_loss/num_samples

# Print results
print("Loss: " + str(dataset_loss))
print("RMSE-loss: " + str(dataset_loss_rmse))
print("Cosine similarity loss: " + str(dataset_loss_cos))
