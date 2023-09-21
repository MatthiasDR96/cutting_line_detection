# Imports
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Encodes a line in rho-theta
def line_encode_1(x1, y1, x2, y2):
	
	# Convert line to Cartesian coordinates
	a = 0 if (x2 - x1) == 0 else (y2 - y1) / (x2 - x1)
	b = y1 - a*x1

	# Compute alfa
	alfa = math.atan2((y2 - y1) , (x2 - x1)) # [-pi, pi] radians
	if alfa < 0: alfa = alfa + math.pi # [0, pi] radians

	# Compute rho
	R = abs(b) / math.sqrt(a**2 + (-1)**2)

	# Compute theta
	if alfa < math.pi/2:
		if b < 0:
			theta = alfa - (math.pi/2)
		else:
			theta = alfa + (math.pi/2) 
	else:
		if b < 0:
			theta = alfa + (math.pi/2) 
		else:
			theta = alfa - (math.pi/2)

	# Set target
	target = np.array([R*math.cos(theta), R*math.sin(theta)])

	return target

# Encodes the line in its two given points
def line_encode_2(x1, y1, x2, y2):

	# Set target
	target = np.array([x1, y1, x2, y2])

	return target

# Cosine similarity loss
def loss_cos(pred, y):
	delta_x_pred = pred[2]-pred[0]
	delta_y_pred = pred[3]-pred[1]
	delta_x_y = y[2]-y[0]
	delta_y_y = y[3]-y[1]
	a = torch.stack((delta_x_pred, delta_y_pred), 0)
	b = torch.stack((delta_x_y, delta_y_y), 0)
	return torch.sum(torch.acos(F.cosine_similarity(torch.abs(a), torch.abs(b), dim=0)) * 180 / math.pi)
 
# Animate
def animate(theta):

	# Simulate a rotating line
	x21 = 100
	y21 = 100
	x22 = x21 + 50*math.cos(theta/180 * math.pi)
	y22 = y21 + 50*math.sin(theta/180 * math.pi)

	# Encode to rho theta
	target_11 = line_encode_1(x11, y11, x12, y12) # Ground truth
	target_12 = line_encode_1(x21, y21, x22, y22) # Predicted

	# Encode to four points
	target_21 = line_encode_2(x11, y11, x12, y12) # Ground truth
	target_22 = line_encode_2(x21, y21, x22, y22) # Predicted
	
	# Cost
	l1_loss = F.l1_loss(torch.tensor(target_12), torch.tensor(target_11), reduction="sum").item() # Computes the L1 loss between the Rho-theta vectors of both lines
	rmse_loss = torch.sqrt(F.mse_loss(torch.tensor(target_12), torch.tensor(target_11),  reduction="sum")).item()  # Computes the Euclidean distance between the Rho-theta vectors of both lines
	cos_sim_loss = loss_cos(torch.tensor(target_22), torch.tensor(target_21)) # Computes the dot product between the two lines (angle in between in degrees)

	# Append
	l1_hist.append(l1_loss)
	rmse_hist.append(rmse_loss)
	cos_sim_hist.append(cos_sim_loss)

	# Plot
	ax.clear()
	ax.plot([x11, x12], [y11, y12])
	ax.plot([x21, x22], [y21, y22])
	ax.plot([0, target_11[0]], [0, target_11[1]], 'b')
	ax.plot([0, target_12[0]], [0, target_12[1]], 'b')
	ax.plot([target_11[0], target_12[0]], [target_11[1], target_12[1]], 'r')
	ax.axis('equal')
	ax.set_xlim(-200, 200)
	ax.set_ylim(-200, 200)

	return ax,

if __name__ == '__main__':

	# Define static line
	x11, y11, x12, y12 = 100, 100, 200, 100

	# Loop over theta
	l1_hist = []
	rmse_hist = []
	cos_sim_hist = []
	
	# Animate
	x = range(0, 360, 1)
	fig, ax = plt.subplots()
	ani = animation.FuncAnimation(fig, animate, repeat=False, frames=len(x) - 1, interval=50)

	# Save as gif
	#writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
	#ani.save('./tests/test_cost.gif', writer=writer)

	# Show
	plt.show()
		
	# Plot cost functions
	plt.plot(range(0, 360, 1), l1_hist)
	plt.plot(range(0, 360, 1), rmse_hist)
	plt.plot(range(0, 360, 1), cos_sim_hist)
	plt.legend(['l1', 'rmse', 'cos_similarity'])
	plt.ylabel('L1 loss - RMSE loss')
	plt.xlabel('Theta [0 - 360] °')
	plt.show()