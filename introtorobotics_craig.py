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
def R_arb_ax(k, theta):
	# theta is in degrees, so make sure it is converted to radians
	phi = theta*np.pi/180
	
	R = np.array([[k[0]**2 * (1 - np.cos(phi)) + np.cos(phi),      k[0]*k[1]*(1 - np.cos(phi)) - k[2]*np.sin(phi), k[0]*k[2]*(1 - np.cos(phi)) + k[1]*np.sin(phi)],
				   [k[0]*k[1]*(1 - np.cos(phi)) + k[2]*np.sin(phi), k[1]**2 * (1 - np.cos(phi)) + np.cos(phi),      k[1]*k[2]*(1 - np.cos(phi)) - k[0]*np.sin(phi)],
				   [k[0]*k[2]*(1 - np.cos(phi)) - k[1]*np.sin(phi), k[1]*k[2]*(1 - np.cos(phi)) + k[0]*np.sin(phi), k[2]**2 * (1 - np.cos(phi)) + np.cos(phi)]])
	
	theta  = np.arccos((R[0,0] + R[1,1] + R[2,2] - 1) / 2)
	k = (1 / (2 * np.sin(theta))) * np.array([[R[2,1] - R[1,2]],
											  [R[0,2] - R[2,0]],
											  [R[1,0] - R[0,1]]])
	# this will need to be tailored to return whatever I need to get out of the rotation matrix - 
	# could need k, could need the angle that it was rotated
	return R


def get_T(k, theta, p):
	R = R_arb_ax(k, theta)
	T = np.eye(4)
	T[:3, :3] = np.reshape(R,(3,3,))
	
	if len(p) != 0:
		T_1 = np.eye(4)
		T_1[:3, 3] = p

		T_1_inv = np.eye(4)
		T_1_inv[:3, 3] = -p

		Tf =  np.mat(T_1) * np.mat(T) * np.mat(T_1_inv)
		return Tf
	else:
		return T


 # Trial code for the above functions
 """
ak = np.array([[0.707], [0.707], [0.0]])
p = np.array([1, 2, 3])

print(R_arb_ax(ak, 30))

print(get_T(ak, 30, p))

v = np.array([0, 0, 1])
theta = 30
t = []
print(get_T(v, theta, t))
"""

# Euler Parameters
def eul_param(k, theta, *R):
	phi = theta * np.pi / 180
	eps1 = k[0] * np.sin(phi / 2)
	eps2 = k[1] * np.sin(phi / 2)
	eps3 = k[2] * np.sin(phi / 2)
	eps4 = np.cos(phi / 2)

	if eps1**2 + eps2**2 + eps3**2 + eps4**2 != 1:
		print("This is WRONG")
		return
	R_eps = np.array([[1 - (2 * eps2**2) - (2 * eps3**2),  2*(eps1*eps2 - eps3*eps4),          2*(eps1*eps3 + eps2*eps4)],
					  [2*(eps1*eps2 + eps3*eps4),          1 - (2 * eps2**2) - (2 * eps3**2),  2*(eps2*eps3 - eps1*eps4)],
					  [2*(eps1*eps3 - eps2*eps4),          2*(eps2*eps3 - eps1*eps4),          1 - (2 * eps2**2) - (2 * eps3**2)]])

#### need to put in an optional parameter to figure out the eps values for if a R is input
# pg 56

