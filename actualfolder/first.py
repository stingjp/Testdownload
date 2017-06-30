# test
import math
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-np.pi,np.pi,num=50)
y = np.sin(x,np.sin(x))

plt.plot(x,y)
plt.xlabel('Angle [rad]')
plt.ylabel('sin(x)')


plt.show()
