import os
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

os.chdir('C:/Users/Will Schulte/Desktop/')
data = np.array(np.genfromtxt('BowlingBall.csv', delimiter=' ', dtype=None, encoding=None))
np.savetxt("BowlingBall_2.csv", data, delimiter=',', fmt='%s')

#fig = plt.figure(figsize=(10,10))
#ax = fig.add_subplot(111, projection='3d')

#for i in range(int(len(data)/10)):
#    ax.scatter(data[i][0],data[i][1],data[i][2])
#    if i%100 == 0:
#        print(i)

#plt.show()