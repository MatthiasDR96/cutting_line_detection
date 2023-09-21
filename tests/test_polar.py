# Imports
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def line_encode(x1, y1, x2, y2):

	# Convert line to polar coordinates
	a = 0 if (x2 - x1) == 0 else (y2 - y1) / (x2 - x1)
	b = y1 - a*x1

	# Compute alfa
	alfa = math.atan2((y2 - y1) , (x2 - x1)) # [-pi, pi] radians
	if alfa < 0: alfa = alfa + math.pi # [0, pi] radians

	# Compute distance
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

	return R*math.cos(theta), R*math.sin(theta)

def line_decode(rcos, rsin):

	# Compute point perpendicular to line from origin
	Sy = rsin
	Sx = rcos
	
	# Apolar is the slope of the perpendicular line. The product of the slopes of 2 perpendicular lines is -1.
	apolar = 0 if Sx == 0 else Sy/Sx
	a = 0 if apolar == 0 else -1/apolar
	b = Sy - a * Sx    

	return a, b

def animate(theta):

    # Simulate a line
    x1 = 100
    y1 = 100
    x2 = x1 + 50*math.cos(theta/180 * math.pi)
    y2 = y1 + 50*math.sin(theta/180 * math.pi)

    # Encode the line
    Sx, Sy = line_encode(x1, y1, x2, y2)
    
    # Decode
    a, b = line_decode(Sx, Sy)

    # Append
    sx_list.append(Sx)
    sy_list.append(Sy)

    # Show
    x = np.linspace(-200, 200, 100)
    ax.clear()
    ax.plot([x1, x2], [y1, y2], 'r.')
    ax.plot([x1, x2], [y1, y2], 'r')
    ax.plot([0, Sx], [0, Sy], 'b')
    ax.plot(x, a*x + b, 'g')
    ax.axis('equal')
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)

    return ax,

if __name__ == '__main__':
	
    # Create hists
    R_list = []
    theta_list = []
    cost_list = []
    sx_list = []
    sy_list = []

    # Animate
    x = range(0, 360, 1)
    fig, ax = plt.subplots()
    ani = animation.FuncAnimation(fig, animate, repeat=False, frames=len(x) - 1, interval=50)

    # Save as gif
    writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('./tests/test_polar.gif', writer=writer)

    # Show
    plt.show()

    # Plot
    plt.plot(range(0, 360, 1), sx_list)
    plt.plot(range(0, 360, 1), sy_list)
    plt.legend(['Sx', 'Sy', 'sin'])
    plt.ylabel('rho*cos(theta) - rho*sin(theta)')
    plt.xlabel('Theta [0 - 360] °')
    plt.show()