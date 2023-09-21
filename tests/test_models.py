# Imports
import pandas as pd
import matplotlib.pyplot as plt

# Model name
model_name = 'model_0.01-16-0-100'

# Plot result plots
results = pd.read_csv('./results/' + model_name + '.csv')

# Plot learning curves
fig = plt.figure(1)
x_epoch = range(1, int(model_name.split('-')[-1]) + 1-3)
plt.plot(x_epoch, results['training_loss'], 'b-', markersize=1, label='training set')
plt.plot(x_epoch, results['validation_loss'], 'r-', markersize=1, label='validation set')
plt.title('Learning curves')
plt.xlabel('Number of epochs')
plt.ylabel('RMSE-loss + cosine similarity loss')
plt.legend()
plt.show()
fig.savefig('./results/' + model_name + '.jpg')

# Plot learning curves rmse
fig = plt.figure(2)
x_epoch = range(1, int(model_name.split('-')[-1]) + 1-3)
plt.plot(x_epoch, results['training_loss_rmse'], 'b-', markersize=1, label='training set')
plt.plot(x_epoch, results['validation_loss_rmse'], 'r-', markersize=1, label='validation set')
plt.title('Learning curves')
plt.xlabel('Number of epochs')
plt.ylabel('RMSE-loss')
plt.legend()
plt.show()
fig.savefig('./results/' + model_name + '_rmse.jpg')

# Plot learning curves cos
fig = plt.figure(3)
x_epoch = range(1, int(model_name.split('-')[-1]) + 1-3)
plt.plot(x_epoch, results['training_loss_cos'], 'b-', markersize=1, label='training set')
plt.plot(x_epoch, results['validation_loss_cos'], 'r-', markersize=1, label='validation set')
plt.title('Learning curves')
plt.xlabel('Number of epochs')
plt.ylabel('Cosine similarity loss')
plt.legend()
plt.show()
fig.savefig('./results/' + model_name + '_cos.jpg')