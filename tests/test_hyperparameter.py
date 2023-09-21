# Imports
import pandas as pd
import matplotlib.pyplot as plt

# Read data
data = pd.read_csv('hyperparameter_tuning.csv', names=['learning_rate', 'batch_size', 'augmentation', 'loss'])

# Plot learning rate
fig = plt.figure(1)
plt.loglog(data['learning_rate'], data['loss'], 'b*', markersize=1)
plt.xlabel('RMSE Loss')
plt.xlabel('Learning rate')
fig.savefig('./results/hyperparameter_tuning_lr.jpg')

# Plot batch size
fig = plt.figure(2)
plt.plot(data['batch_size'], data['loss'], 'b*', markersize=1)
plt.xlabel('RMSE Loss')
plt.xlabel('Bacth size')
fig.savefig('./results/hyperparameter_tuning_bs.jpg')

# Plot augmentation
fig = plt.figure(3)
plt.plot(data['augmentation'], data['loss'], 'b*', markersize=1)
plt.xlabel('RMSE Loss')
plt.xlabel('Augmentation')
fig.savefig('./results/hyperparameter_tuning_aug.jpg')
plt.show()