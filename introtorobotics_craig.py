# introtorobotics_craig.py
import numpy as np
"""
R = [[0.8660254, -0.5, 0], [0.5, 0.8660254, 0], [0, 0, 1]]
print(np.linalg.det(R))
print(-.5*.866)
"""

"""
for i in range(0, np.pi, np.pi / 3):
	print(np.cos(i+90))
	print(np.sin)
print(np.cos(2*np.pi/3))
print(np.sin(np.pi/6))
	"""

# create some rotation matrices
def Rx(gamma):
	return np.array([[1.,            0.,             0.],
					[0., np.cos(gamma), -np.sin(gamma)],
					[0., np.sin(gamma),  np.cos(gamma)]])

def Ry(beta):
	return np.array([[ np.cos(beta), 0., np.sin(beta)],
					[           0., 1., 		  0.],
					[-np.sin(beta), 0., np.cos(beta)]])

def Rz(alpha):
	return np.array([[np.cos(alpha), -np.sin(alpha), 0.],
				    [np.sin(alpha),  np.cos(alpha), 0.],
				    [           0.,	            0.,	1.]])

def Rzyz(angles):
	if len(angles) == 3:
		a, b, c = angles
	elif len(angles) == 2:
		a, b == angles
		c = -a
	return np.array([[np.cos(a)*np.cos(b)*np.cos(c) - np.sin(a)*np.sin(c), -np.cos(a)*np.cos(b)*np.sin(c) - np.sin(a)*np.cos(c), np.cos(a)*np.sin(b)],
					 [np.sin(a)*np.cos(a)*np.cos(c) + np.cos(a)*np.sin(c), -np.sin(a)*np.cos(b)*np.sin(c) + np.cos(a)*np.cos(c), np.sin(a)*np.sin(b)],
					 [								 -np.sin(b)*np.cos(c),  								np.sin(b)*np.sin(c), 		   np.cos(b)]])


def get_zyz_angles(R):
	b = np.arctan2(np.sqrt(R[2,0]**2 + R[2,1]**2), R[2,2])
	if b == 0:
	   	a = 0
	   	c = np.arctan2(-R[0, 1], R[0, 0])
	elif b == np.pi:
		a = 0
		c = np.arctan2(R[0,1], -R[0,0])
	else:
		a = np.arctan2(R[1,2] / np.sin(b), R[0,2] / np.sin(b))
		c = np.arctan2(R[2,1] / np.sin(b), -R[2,0] / np.sin(b))
	return np.array([a, b, c])

# Equivalent angle axis
def arbitrary_axis_rot(k, R, theta):
	# theta is in degrees, so make sure it is converted to radians
	phi = theta*np.pi/180
	Rk = np.array([[k[0]**2 * (1 - np.cos(phi)) + np.cos(phi),      k[0]*k[1]*(1 - np.cos(phi)) - k[2]*np.sin(phi), k[0]*k[2]*(1 - np.cos(phi)) + k[1]*np.sin(phi)],
				   [k[0]*k[1]*(1 - np.cos(phi)) + k[2]*np.sin(phi), k[1]**2 * (1 - np.cos(phi)) + np.cos(phi),      k[1]*k[2]*(1 - np.cos(phi)) - k[0]*np.sin(phi)],
				   [k[0]*k[2]*(1 - np.cos(phi)) - k[1]*np.sin(phi), k[1]*k[2]*(1 - np.cos(phi)) + k[0]*np.sin(phi), k[2]**2 * (1 - np.cos(phi)) + np.cos(phi)]])
	
	theta  = np.arccos((R[0,0] + R[1,1] + R[2,2] - 1) / 2)
	k = (1 / (2 * np.sin(theta))) * np.array([[R[2,1] - R[1,2]],
											  [R[0,2] - R[2,0]],
											  [R[1,0] - R[0,1]]])
	# this will need to be tailored to return whatever I need to get out of the rotation matrix - could need k, could need the angle that it was rotated

	return

	