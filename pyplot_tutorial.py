# pyplot_tutorial.py

import matplotlib.pyplot as plt 			# need it
import matplotlib.animation as animation 	
from matplotlib import style
import numpy as np 							# need it
from random import randint 					
from mpl_toolkits.mplot3d import axes3d 	# need it
import math 								# need it
import seaborn as sns

"""
t = np.arange(0.0, 2.0, 0.01)
s1 = np.sin(2*np.pi*t)
s2 = np.sin(4*np.pi*t)

"""
# using the subplot call, you can set up a figure with multiple plots on it in
# an n x n matrix format, with the third number being the location. 
# top left = 1, top right = 2
# bottom left = 3, bottom right = 4
"""

plt.figure(1)
plt.subplot(221)
line1 = plt.plot(t, s1, 'b8', alpha=0.1, label='1')
plt.legend()
plt.grid(True)
"""
# alpha is a parameter for the transparency, 0-1, 0 being transparent.
"""
plt.subplot(222)
line2 = plt.plot(t, 2*s1, 'y^', label='2')
plt.grid(True)
plt.legend()

plt.subplot(223)
line3 = plt.plot(t, s2, 'k1', label='3')
plt.grid(True)
plt.legend()

"""
# this one has all of the bells and whistles!!
"""

plt.subplot(224)
line4 = plt.plot(t, s2, 'g_', label='4')
plt.grid(True)
plt.xlabel('plot #4')
plt.ylabel('metric')
plt.title('Title\nCheck it out')
plt.legend()
ax = plt.gca()
ax.set_xticklabels([])

# this one shows just a single figure with one graph that has the legend
plt.figure(2)
plt.plot(t, s1, 'r.', label='first')
plt.plot(t, s2, 'ko', label='second')
plt.legend(loc='upper right')

# plt.show()

""" 
# on to scatters
"""
x = [1,2,3,4,5,6,7,8,9]
y = [5,2,7,5,9,2,1,7.5,8]

plt.figure(3)
plt.scatter(x, y, label='scatter', color='k', marker='^', s=0.3)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('title')

"""
# on to live graphs.... maybe...
# this is useful for constantly updating data sets i.e. a text file that is 
# updated and saved over time
"""

style.use('fivethirtyeight')

fig4 = plt.figure(4)
ax4 = fig4.add_subplot(111)

def animate(i):
	graph_data = open('samplefile.txt', 'r').read()
	lines = graph_data.split('\n')
	x4 = []
	y4 = []
	for line in lines:
		if len(line) > 1:
			x, y = line.split(',')
			x4.append(x)
			y4.append(y)
		ax4.clear()
		ax4.plot(x4, y4)			

ani = animation.FuncAnimation(fig4, animate, interval=1000)
"""
# alright starting into 3d

fig5 = plt.figure(5)
ax5 = fig5.add_subplot(111, projection='3d')

x5 = list(range(1,1000))
y5 = [math.log(i, 2) for i in x5]
z5 = [math.sin(i/10) for i in x5]

ax5.plot_wireframe(x5, y5, z5)
ax5.set_xlabel('x axis')
ax5.set_ylabel('y axis')
ax5.set_zlabel('z axis')


# 3D scatter

fig6 = plt.figure(6)
ax6 = fig6.add_subplot(111, projection='3d')

x6 = (list(range(1, 1000)))
y6 = [math.log(i, 2) for i in x6]
z6 = [math.sin(i/10) for i in x6]

ax6.scatter(x6, y6, z6, c='r')
ax6.set_xlabel('x axis')
ax6.set_ylabel('y axis')
ax6.set_zlabel('z axis')

 
# definitely all useful
"""

style.use('fivethirtyeight')

fig7 = plt.figure(7)
ax7 = fig7.add_subplot(111, projection='3d')

x7, y7, z7 = axes3d.get_test_data()
print(axes3d.__file__)
ax7.plot_wireframe(x7, y7, z7, rstride=3, cstride=3)

ax7.set_xlabel('x axis')
ax7.set_ylabel('y axis')
ax7.set_zlabel('z axis')

plt.show()
"""
#sns.figure(8)
sns.set(style='darkgrid', palette='Set2')
"""
# create a noisy periodic data set
sines = []
rs = np.random.RandomState(8)
for _ in range(15):
	x = np.linspace(0, 30 / 2, 30)
	y = np.sin(x) + rs.normal(0, 1.5) + rs.normal(0, .3, 30)
	sines.append(y)

# plot th eaverage over replicates with bootstrap resamples
sns.tsplot(sines, err_style='boot_traces', n_boot=500)
"""

x = list(range(1,1000))
y = [math.log(i, 2) for i in x]
z = [math.sin(i/10) for i in x]


# sns.tsplot(x, y, z)
plt.plot(x, y, z)

sns.plt.show()
