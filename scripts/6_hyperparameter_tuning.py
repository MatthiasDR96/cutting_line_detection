# Imports
import torch
from utils import *
import pandas as pd
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader

# Check if there is a graphical card (CUDA) available on the PC
device = torch.device("cuda:" + str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")

# File paths
root_folder = './data'

# Number of epochs
epochs = 50

# Load datasets
df_train = pd.read_pickle(root_folder + '/train_data.csv')
df_valid = pd.read_pickle(root_folder + '/valid_data.csv')

# Train model
best_loss = float('inf')
best_model = []
best_results = []
best_learning_rate = 0
best_batch_size = 0
best_augmentation = 0
lst = []
for learning_rate in [0.1, 0.01, 0.001, 0.0001]:
	for batch_size in [16, 32, 64]:
		for augmentation in [0, 1]:

			# Debug
			print('Learning rate - ' + str(learning_rate) + ' - ' + 'Batch size - ' + str(batch_size) + ' - ' + 'Augmentation - ' + str(augmentation))

			# Create datasets
			train_ds = Dataset(df_train['path_names'], transform=bool(augmentation))
			valid_ds = Dataset(df_valid['path_names'], transform=False)

			# Create data loaders
			train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
			valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
			dataloaders = {'train': train_dl, 'valid': valid_dl}

			# Reset last layers
			model = models.resnet18(pretrained=True)
			total_features = model.fc.in_features
			model.fc = nn.Linear(total_features, 4)
			model.to(device)

			# Set optimizer
			optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
						
			# Train model
			loss, loss_rmse, loss_cos, model, results = train_model(model, optimizer, dataloaders, device, epochs)

			# Save hyperparameter results
			lst.append([learning_rate, batch_size, augmentation, loss, loss_rmse, loss_cos])
			df = pd.DataFrame(lst, columns=['learning_rate', 'batch_size', 'augmentation', 'loss', 'loss_rmse', 'loss_cos'])
			df.to_csv('./results/hyperparameter_tuning.csv', index=False)
						
			# Select best model
			if loss < best_loss:
				best_loss = loss
				best_model = model
				best_results = results
				best_learning_rate = learning_rate
				best_batch_size = batch_size
				best_augmentation = augmentation

# Print results
print("Best learning rate: " + str(best_learning_rate))
print("Best batch size: " + str(best_batch_size))
print("Best augmentation: " + str(best_augmentation))

# Model name
model_name = "best_model"

# Save model
torch.save(best_model, './models/' + model_name + '.pt')

# Save results
df = pd.DataFrame(best_results)
df.to_csv('./results/' + model_name + '.csv', index=False)