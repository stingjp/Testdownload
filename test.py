import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal

# try:
#     for i in range(100):
#         print(i)
#         time.sleep(1)
# except KeyboardInterrupt:
#     pass
# print('done, save now')


# import socket;
# socket.socket(socket.AF_INET,socket.SOCK_STREAM).connect(("localhost",52265))

filename = r"C:\\Users\\JP\\Google Drive\\code\\python\\heart_sjm_1x3.p"
with open(filename, 'rb') as f:
    heart_1x3 = pickle.load(f)
# ascension data
filename = r"C:\\Users\\JP\\Google Drive\\code\\python\\ascension_test_1x3.p"
with open(filename, 'rb') as f:
    asc_1x3 = pickle.load(f)

# print(heart_1x3.shape)
# print(asc_1x3.shape)

a1_1x3 = asc_1x3[0, :, :, :]
a2_1x3 = asc_1x3[1, :, :, :]
a3_1x3 = asc_1x3[2, :, :, :]
a_1x3  = np.concatenate((a1_1x3, a2_1x3, a3_1x3), axis=0)
# print(a_1x3.shape)


f1 = plt.figure(1)

a1 = f1.add_subplot(111)
a1.plot(a_1x3[:, 1, 0])

# this is where you look at the actual data sets and select points from the ascenstion
# to put as the initial index - lines all of the data up
slice_dist = 150
slice_asc = []
slice_asc.append(0)
slice_asc.append(205)
slice_asc.append(410)

# this just looks at the slice point and plots a vertical line at that point
for slice_pt in slice_asc:
    plt.plot(slice_pt * np.ones(2), [50, 300])
    plt.plot((slice_pt +slice_dist ) * np.ones(2), [50, 300], '--')

# my individual plots of the data
f1 = plt.figure()
a1 = f1.add_subplot(111)
a1.plot(heart_1x3[:, 0, 0], c='r')
a1.plot(heart_1x3[:, 0, 1], c='orange')
a1.plot(heart_1x3[:, 0, 2], c='g')

# here is the same slicing from the sjm data
slice_dist = 150
slice_pts = []
slice_pts.append(550)
slice_pts.append(1266)
slice_pts.append(2020)

# plots the vertical lines on the sjm data. 
for slice_pt in slice_pts:
    plt.plot(slice_pt * np.ones(2), [-300, 300])
    plt.plot((slice_pt +slice_dist ) * np.ones(2), [-300, 300], '--')
    # print(slice_pt)

# This gets the number of stops (angle places that I stopped at in the experiment for the given 
# orientation)
n_stops = len(slice_pts)
# this is the number of data points selected at each stop (150 chosen)
n_data_pts = slice_dist
# 4 electrodes + ascension
n_sensors = 5
# x, y, z, alpha, beta, gamma
n_DOF = 6

# inputs all of the parameters that we will have into one single matrix
data = np.zeros((n_stops, n_data_pts, n_sensors, n_DOF))

# print(data)
# print(data.shape)

# this adds the data from the file into the major matrix
# the first dimension is the stops, which is essentially what we are 
# iterating through with the slice points

# ??? [slice_pt:slice_pt] ??? how does that work???

for i, slice_pt in enumerate(slice_pts):
    data[i, :, :4, :3] = heart_1x3[slice_pt:slice_pt + slice_dist]
# print(data)

# same as the last function, but for the ascension sensor to be added to the
# major matrix

for i, slice_pt in enumerate(slice_asc):
    data[i, :, 4, :] = a_1x3[slice_pt:slice_pt + slice_dist, 1, :6]
# print(data)

# data = np.zeros((n_stops, n_data_pts, n_sensors, n_DOF))


plt.figure()
default_colors = ['k', 'b', 'r', 'g', 'c']
avg_dist = []
for i_sensor in range(n_sensors):
    for i_pt in range(n_stops):
        plt.plot(data[i_pt,:,i_sensor,0], data[i_pt,:,i_sensor,1], color = default_colors[i_sensor])
        plt.plot(np.mean(data[i_pt,:,i_sensor,0]), np.mean(data[i_pt,:,i_sensor,1]), 'o')
 
plt.xlabel('X Position')
plt.ylabel('Y Position')
# for each stop, and each sensor, plot the whole set of data (x, y)
# plot the average of all the x and y values for each stop, for each sensor

av_stops = len(slice_pts)
av_data_pts = 1
av_sensors = 5
av_DOF = 6

av_data = np.zeros((av_stops, av_data_pts, av_sensors, av_DOF))

# print(av_data.shape)

for i_sensor in range(av_sensors):
    for i_stop in range(av_stops):
        for i_dim in range(av_DOF):
            # print(np.mean(data[i_stop, :, 0, i_dim]))
            av_data[i_stop, :, i_sensor, i_dim] = np.mean(data[i_stop, :, i_sensor, i_dim])
            # print(av_data)

#print(av_data.shape)
# print(av_data)

# data = np.zeros((n_stops, n_data_pts, n_sensors, n_DOF))
test_set1 = av_data[0, 0, :, :]
# print(test_set1.shape)
# print(test_set1)

# I am going to multiply to the ascension dimensions - scale is 10*SJM = ASC

sjm_test = 10 * test_set1[:4, :3]
asc_test = test_set1[4, :3]

print(sjm_test.shape)
print(asc_test.shape)

# a^-1 = a / |a|^2


for i in range(4):
    s = sjm_test[i, :]
    inv_s = s / ((np.absolute(s))**2)
    T = inv_s * asc_test
    print(T.shape)
    print(T)

# I'm honestly not super sure if this works or anything. The transformation seems to be fairly consistent
# for the second and third dimension, but the first is not great at all.     
