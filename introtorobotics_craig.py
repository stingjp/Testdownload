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
	return np.array([1.,            0.,             0.],
					[0., np.cos(gamma), -np.sin(gamma)],
					[0., np.sin(gamma),  np.cos(gamma)])

def Ry(beta):
	return np.array([ np.cos(beta), 0., np.sin(beta)],
					[           0., 1., 		  0.],
					[-np.sin(beta), 0., np.cos(beta)])

def Rz(alpha):
	return np.array([np.cos(alpha), -np.sin(alpha), 0.],
				    [np.sin(alpha),  np.cos(alpha), 0.],
				    [           0.,	            0.,	1.])




aK = np.array[[0.707], [0.707], [0.0]]
