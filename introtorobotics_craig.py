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
# these are another way in which to describe a frame or position
def eul_param(k, theta, *R):
	phi = theta * np.pi / 180
	if R is None:
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
	elif R is not None:
		eps4 = (1/2)*np.sqrt(1 + R[0,0] + R[1,1] + R[2,2])
		eps3 = (R[0,2] - R[2,0]) / (4 * eps4)
		eps2 = (R[1,0] - R[0,1]) / (4 * eps4)
		eps1 = (R[2,1] - R[1,2]) / (4 * eps4)
		print("Note that if the rotation was 180 degrees, these will not be useful")


#### need to put in an optional parameter to figure out the eps values for if a R is input
# pg 56

##############################################################################
# here I will begin the chapter 2 exercises. 
# programming exercises


def mulitply_Ts(ArelB, BrelC, *args):
	T_int = np.mat(ArelB) * np.mat(BrelC)
	if args is not None:
		for each in args:
			T_int = T_int * np.mat(each)
		return T_int
	else:
		return T_int


def inv_T(T_1):
	T_T = np.transpose(T_1)
	return T_T



gh = np.array([[1, 2, 3],
			   [4, 5, 6],
			   [0, 0, 0]])

g = inv_T(gh)
print(mulitply_Ts(g, gh))

##############################################################################
# stuff from trig.py - soft_robots - Jake Sganga
'''
Collection of all the trig functions / rotation matrices that are used throughout
the simulation and control functions
'''
import numpy as np

def Rx(alpha):
    return np.array([[1, 0, 0.],
                    [0,   np.cos(alpha), np.sin(alpha)],
                    [0.,   -np.sin(alpha), np.cos(alpha)]])

def Rz(alpha):
    return np.array([[np.cos(alpha), -np.sin(alpha), 0.],
                    [np.sin(alpha),   np.cos(alpha), 0.],
                    [           0.,              0., 1.]])
def Ry(alpha):
    return np.array([[np.cos(alpha), 0., np.sin(alpha)],
                    [            0., 1., 0.],
                    [-np.sin(alpha), 0., np.cos(alpha)]])

'''
R_zyz from Craig pg 49. See README figure
a = alpha (+Z, ground), b = beta (+Y'), c = gamma (+Z")
ranges:
a: [-pi, pi]
b: [0, pi]
c: [-pi, pi]
arctan2 automatically enforces the -pi to pi bounds
the sqrt on the eqn for b (see get_zyz_angles), enforces [0, pi]
note that care needs to be taken with ranges and delta angles

 Elevation = -pi/2 + beta *************
'''

def R_zyz(angles):
    if len(angles) == 3:
        a, b, c = angles
    elif len(angles) == 2: #often used for cath rotations, allows seemless replacement of RzRyRz
        a, b = angles
        c = -a
    return np.array([
        [np.cos(a) * np.cos(b) * np.cos(c) - np.sin(a) * np.sin(c), -np.cos(a) * np.cos(b) * np.sin(c) - np.sin(a) * np.cos(c), np.cos(a) * np.sin(b)],
        [np.sin(a) * np.cos(b) * np.cos(c) + np.cos(a) * np.sin(c), -np.sin(a) * np.cos(b) * np.sin(c) + np.cos(a) * np.cos(c), np.sin(a) * np.sin(b)],
        [                                   -np.sin(b) * np.cos(c),                                      np.sin(b) * np.sin(c),             np.cos(b)]])

def get_zyz_angles(R):
    b = np.arctan2(np.sqrt(R[2,0]**2 + R[2,1]**2), R[2,2])
    if b == 0:
        a = 0
        c = np.arctan2(-R[0,1], R[0,0])
    elif b == np.pi:
        a = 0
        c = np.arctan2(R[0,1], -R[0,0])
    else:
        a = np.arctan2(R[1,2] / np.sin(b), R[0,2] / np.sin(b))
        c = np.arctan2(R[2,1] / np.sin(b), -R[2,0] / np.sin(b))
    return np.array([a, b, c])



def aer_to_abc(sensor_angles):
    # converting ascension angles (degress) to euler angles (radians)
    # R_zyz expects abc form
    # elevation = -pi/2 + beta
    a, el_rad, c = sensor_angles * np.pi / 180
    b =  el_rad + np.pi / 2
    return np.array([a, b, c])

def abc_to_aer(euler_angles):
    az, b_deg, roll = euler_angles * 180 / np.pi
    el = b_deg - 90.
    return np.array([az, el, roll])
    

def get_T_from_x(X):
    '''
    The angles can be described by the rotation
    matrices Rz(alpha)Ry(beta)Rz(-azimuth) according to a intrinsic rotation
    (first about Z by azimuth, then about the resulting -Y axis by elevation, then correcting
    for the roll introduced) - called R_zyz(alhpa, beta, gamma) where gamma = -alpha 
    Note still using radians
    '''
    x, y, z, alpha, beta, gamma = X
    R = R_zyz([alpha, beta, gamma])
    d = np.array([x, y, z])
    T = np.eye(4)
    T[:3,:3] = R[:]
    T[:3, 3] = d 
    return T

def get_x_from_T(T):
    '''
    This function will extract the x, y, z, azimuth, elevation in the corrected
    coordinate frame (x = insertion) to match the ascension setup
    also makes the azimuth term more stable for most of the workspace
    '''
    x, y, z = T[:3,3]
    alpha, beta, gamma = get_zyz_angles(T[:3,:3])
    return np.array([x, y, z, alpha, beta, gamma])

def get_lookat_angles(lookat_point, current_point):
    '''
    gets alpha and beta (R_zyz) for a vector from the current point
    to the lookat point. 
    '''
    dir_look_at   = normalize(lookat_point - current_point)
    a             = np.arctan2(dir_look_at[1], dir_look_at[0])
    b             = np.arccos(dir_look_at[2])
    return a, b

def get_vec_length(vec):
    # handles 1D and 2D arrays
    # measures lengths of 2D rows
    if vec.ndim == 1:
        return np.sqrt(vec.dot(vec))
    elif vec.ndim == 2:
        return np.sqrt(np.sum(np.square(vec), axis = 1))

def normalize(vec):
    # handles 1D and 2D arrays
    if vec.ndim == 1:
        norm = get_vec_length(vec)
        if norm:
            return vec / norm
        else:
            return vec
            
    elif vec.ndim == 2:
        return np.array([x / y for x, y in zip(vec, get_vec_length(vec))])


############# EKF rotations and derivatives #####################
'''
EKF relies on linearizing a non-linear function about the point
of the current state estimate. In this case, the non-linear function is
our expected dx (dx_expected), which will be compared to dx_observed 
in the measurement update step: res = dx_observed - h([a,b,c])
h([a,b,c]) = R_zyx(a,b,c) * J(q) * dq = R_zyx(a,b,c) * dx_model = dx_expected
where a, b, c are the Euler angles yaw, pitch, roll, respectively.
See ModelEnvironment.py for the matrix R_zyx, which is the transpose
of the matrix in Craig pg. 49.
To linearize, we want a jacobian matrix such that h([a,b,c]) ~ H [a, b, c]'
H = dh/d[a, b, c]
This gets a little tricky because we have a 3x3 matrix of non-linear functions 
mutiplied by dx_model to give us our output. 
Below I take the derivative of each row of R_zyx(a,b,c) with respect to a, b, c, 
this requires some transposing to make everything work out.
for example, in the x direction:
h([a,b,c]) ~= H [a, b, c]'
H = [
    [dh_x/da, dh_x/db, dh_x/dc],
    [dh_y/da, dh_y/db, dh_y/dc],
    [dh_z/da, dh_z/db, dh_z/dc]
]

[dh_x/da, dh_x/db, dh_x/dc] = (dRx_dabc * dx_model)'
'''
def R_zyx(angles):
    '''
    Forms rotation matrix based on azimuth elevation and roll anlges
    performing yaw (+z) first (around global z), which is typically called alpha = a
    then pitch (+y) (around relative y), aka beta = b, note that this is different that elevation, used elsewhere
    then roll  (+x) (around most relative x), aka gamma = c
    Z-Y-X Euler angles goes from rotated frame back to ground
    We want to multiply the ground into the rotated frame, so taking transpose of Rzyx 
    in page 49 of Craig
    '''
    a, b, c = angles
    return np.array([
        [np.cos(a) * np.cos(b), np.sin(a) * np.cos(b), -np.sin(b)],
        [np.cos(a) * np.sin(b) * np.sin(c) - np.sin(a) * np.cos(c), np.sin(a) * np.sin(b) * np.sin(c) + np.cos(a) * np.cos(c), np.cos(b) * np.sin(c)],
        [np.cos(a) * np.sin(b) * np.cos(c) + np.sin(a) * np.sin(c), np.sin(a) * np.sin(b) * np.cos(c) - np.cos(a) * np.sin(c), np.cos(b) * np.cos(c)]])

def getABCfromR(R):
    # assumes matrix in form of Rzyx from Craig pg 49
    a = np.arctan2(R[0, 1], R[1, 1])
    b = -np.arcsin(R[0, 2])
    c = np.arctan2(R[1, 2], R[2, 2])
    return np.array([a, b, c])


def dRx_dabc(angles):
    '''
    Derivative of the x row of R_zyx
    Note that a, b, c represent the Euler angles yaw, pitch, roll - axes(Z, Y, X)
    Rx = [np.cos(a) * np.cos(b), np.sin(a) * np.cos(b), -np.sin(b)]
    '''
    a, b, c = angles
    return np.array([
        [ -np.sin(a) * np.cos(b),    np.cos(a) * np.cos(b),          0], #da
        [np.cos(a) *(-np.sin(b)), np.sin(a) * (-np.sin(b)), -np.cos(b)], #db
        [                      0,                        0,          0]]) # dc

def dRy_dabc(angles):
    '''
    Derivative of the y row of R_zyx
    Note that a, b, c represent the Euler angles yaw, pitch, roll - axes(Z, Y, X)
    Ry = [np.cos(a) * np.sin(b) * np.sin(c) - np.sin(a) * np.cos(c), np.sin(a) * np.sin(b) * np.sin(c) + np.cos(a) * np.cos(c), np.cos(b) * np.sin(c)]
    '''
    a, b, c = angles
    return np.array([
        [(-np.sin(a)) * np.sin(b) * np.sin(c) - np.cos(a) * np.cos(c), np.cos(a) * np.sin(b) * np.sin(c) + (-np.sin(a)) * np.cos(c), 0], #da
        [np.cos(a) * np.cos(b) * np.sin(c),                            np.sin(a) * np.cos(b) * np.sin(c),                            (-np.sin(b)) * np.sin(c)], #db
        [np.cos(a) * np.sin(b) * np.cos(c) - np.sin(a) * (-np.sin(c)), np.sin(a) * np.sin(b) * np.cos(c) + np.cos(a) * (-np.sin(c)), np.cos(b) * np.cos(c)]]) # dc

def dRz_dabc(angles):
    '''
    Derivative of the z row of R_zyx
    Note that a, b, c represent the Euler angles yaw, pitch, roll - axes(Z, Y, X)
    Rz = [np.cos(a) * np.sin(b) * np.cos(c) + np.sin(a) * np.sin(c), np.sin(a) * np.sin(b) * np.cos(c) - np.cos(a) * np.sin(c), np.cos(b) * np.cos(c)]
    '''
    a, b, c = angles
    return np.array([
        [(-np.sin(a)) * np.sin(b) * np.cos(c) + np.cos(a) * np.sin(c), np.cos(a) * np.sin(b) * np.cos(c) - (-np.sin(a)) * np.sin(c), 0], #da
        [np.cos(a) * np.cos(b) * np.cos(c)                           , np.sin(a) * np.cos(b) * np.cos(c)                           , (-np.sin(b)) * np.cos(c)], #db
        [np.cos(a) * np.sin(b) * (-np.sin(c)) + np.sin(a) * np.cos(c), np.sin(a) * np.sin(b) * (-np.sin(c)) - np.cos(a) * np.cos(c), np.cos(b) * (-np.sin(c))]]) # dc


########## from https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/10-Unscented-Kalman-Filter.ipynb
# used to handle angles as state variables
def normalize_angle(x):
    x = x % (2 * np.pi)    # force in range [0, 2 pi)
    if x > np.pi:          # move to [-pi, pi)
        x -= 2 * np.pi
    return x


def state_mean(sigmas, Wm):
    # Calculates mean of angles (handles the 359 and 3 deg mean = 1 deg)
    x = np.zeros(3)

    sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
    sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))
    x[0] = np.sum(np.dot(sigmas[:, 0], Wm))
    x[1] = np.sum(np.dot(sigmas[:, 1], Wm))
    x[2] = atan2(sum_sin, sum_cos)
    return x

def z_mean(sigmas, Wm):
    z_count = sigmas.shape[1]
    x = np.zeros(z_count)

    for z in range(0, z_count, 2):
        sum_sin = np.sum(np.dot(np.sin(sigmas[:, z+1]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:, z+1]), Wm))

        x[z] = np.sum(np.dot(sigmas[:,z], Wm))
        x[z+1] = atan2(sum_sin, sum_cos)
    return x

def correct_delta_angles(angles):
    a, b, c = angles
    # a and c are put in [-pi, pi)
    a = a % (2 * np.pi)    # range [0, 2 pi)
    if a > np.pi:          # move to [-pi, pi)
        a -= 2 * np.pi

    b = b % (np.pi)    # range [0, 2 pi)
    if b > np.pi:          # move to [0, pi]
        b -= 2 * np.pi

    c = c % (2 * np.pi)    # range [0, 2 pi)
    if c > np.pi:          # move to [-pi, pi)
        c -= 2 * np.pigit

