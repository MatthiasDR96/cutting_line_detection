### This script trains a neural network using the parameters set in the config folder. To debug, the first image of each 
# batch can be plot with the ground thruth line and the predicted line. This can be uncommented in utils.

# Imports
import json
import torch
from utils import *
import pandas as pd
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader

# Check if there is a graphical card (CUDA) available on the PC
device = torch.device("cuda:" + str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")

# Load training parameters
with open("./config/parameters.json") as f: par = json.load(f)

# Load datasets
df_train = pd.read_pickle(par['training_data_dir'] + '/train_data.csv')
df_valid = pd.read_pickle(par['training_data_dir'] + '/valid_data.csv')

# Create datasets
train_ds = Dataset(df_train['path_names'], transform=bool(par["augmentation"]))
valid_ds = Dataset(df_valid['path_names'], transform=False)

# Create data loaders
train_dl = DataLoader(train_ds, batch_size=par['batch_size'], shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=par['batch_size'], shuffle=True)
dataloaders = {'train': train_dl, 'valid': valid_dl}

# Reset last layer
num_features = 4 # The model predicts two points [x1, y1, x2, y2]
model = models.resnet18(pretrained=True)
total_features = model.fc.in_features
model.fc = nn.Linear(total_features, num_features)
model.to(device)

# Model name
model_name = "model_" + str(par['learn_rate']) + "-" + str(par["batch_size"]) + '-' + str(par["augmentation"]) + '-' + str(par["epochs"])

# Set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=par['learn_rate'])

# Train model
loss, loss_rmse, loss_cos, model, results = train_model(model, optimizer, dataloaders, device, epochs=par["epochs"])

# Print results
print("Loss: " + str(loss))
print("RMSE-loss: " + str(loss_rmse))
print("Cosine similarity loss: " + str(loss_cos))

# Save model
torch.save(model, './models/' + model_name + '.pt')

# Save results
df = pd.DataFrame(results)
df.to_csv('./results/' + model_name + '.csv', index=False)

# Plot learning curves
fig = plt.figure(1)
x_epoch = range(1, par["epochs"]+1)
plt.plot(x_epoch, results['training_loss'], 'b-', markersize=1, label='train')
plt.plot(x_epoch, results['validation_loss'], 'r-', markersize=1, label='valid')
plt.xlabel('Number of epochs')
plt.ylabel('RMSE-loss')
plt.legend()
fig.savefig('./results/' + model_name + '.jpg')