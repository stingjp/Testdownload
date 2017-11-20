# import numpy as np
# import time
# import pickle
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy import signal

# # try:
# #     for i in range(100):
# #         print(i)
# #         time.sleep(1)
# # except KeyboardInterrupt:
# #     pass
# # print('done, save now')


# # import socket;
# # socket.socket(socket.AF_INET,socket.SOCK_STREAM).connect(("localhost",52265))

# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy import signal

# # heart data
# filename = r"C:\\Users\\JP\\Google Drive\\code\\python\\heart_sjm_4x3.p"
# with open(filename, 'rb') as f:
#     heart_4x3 = pickle.load(f)
# # ascension data
# filename = r"C:\\Users\\JP\\Google Drive\\code\\python\\ascension_test_4x3.p"
# with open(filename, 'rb') as f:
#     asc_4x3 = pickle.load(f)


    # Ascension data into vertical arrays
# x_asc_small = np.zeros((24))
# y_asc_small = np.zeros((24))
# z_asc_small = np.zeros((24))
# count = 0
# for i_stop in range(6):
#     for i_sensor in range(n_electrodes):
#         x_asc_small[count] = new_ascension[i_stop, i_sensor, 0]
#         y_asc_small[count] = new_ascension[i_stop, i_sensor, 1]
#         z_asc_small[count] = new_ascension[i_stop, i_sensor, 2]
#         count += 1


# all of the coordinates in one
# xyz_asc_small = np.zeros((24, 3))
# xyz_asc_small[:, 0] = x_asc
# xyz_asc_small[:, 1] = y_asc
# xyz_asc_small[:, 2] = z_asc

# y = Ax
# y values will be the ascension dimensional value x, y, z for each of the sensors
# x values will be the sjm units for each of the dimensions

# need to build the A matrix
# x, x**2, y, y**2, z, z**2, xy, yz, xz
# all St. Jude data
# x_small = sjm_all_rotated[0, :24]
# y_small = sjm_all_rotated[1, :24]
# z_small = sjm_all_rotated[2, :24]



# x = np.zeros((48))
# y = np.zeros((48))
# z = np.zeros((48))
# count = 0
# for i_stop in range(n_stops):
#     for i_sensor in range(n_electrodes):
#         x[count] = av_data[i_stop, i_sensor, 0]
#         y[count] = av_data[i_stop, i_sensor, 1]
#         z[count] = av_data[i_stop, i_sensor, 2]
#         count += 1

# this is the large case
# A_small = np.ones((24, 10))
# A_small[:, 0] = x
# A_small[:, 1] = x**2
# A_small[:, 2] = y
# A_small[:, 3] = y**2
# A_small[:, 4] = z
# A_small[:, 5] = z**2
# A_small[:, 6] = x*y
# A_small[:, 7] = y*z
# A_small[:, 8] = x*z


# large case analysis

# x_all_coef_A_small = np.linalg.pinv(A_small).dot(xyz_asc_small)
# print(x_small_A_small)

####################################################################

# r = A.dot(x_all_coef_A_small) - xyz_asc
# print(np.std(r))
# print(np.mean(r))


# points = np.zeros((48, 3))
# for i in range(24):#len(xyz_asc)):
#     points[i] = A[i].dot(x_all_coef_A)

# dist = np.zeros((len(xyz_asc)))
# for i in range(48):#len(xyz_asc)):
#     dist[i] = np.linalg.norm(A[i].dot(x_all_coef_A) - xyz_asc[i])
# print("standard dev. of norm is:   %f" % np.std(dist))
# print("mean of the norm is:        %f" % np.mean(dist))



######################################################################

# plt.figure()
# for i in range(len(xyz_asc)):
#     plt.plot(xyz_asc[i, 0], xyz_asc[i, 1], color='r', marker='v')
#     plt.plot(points[i, 0], points[i, 1], color='b', marker='.')
# plt.xlabel('X Axis\nRed: Ascension    Blue: Estimates')
# plt.ylabel('Y axis')

# plt.figure()
# plt.plot(dist, 'o')    




