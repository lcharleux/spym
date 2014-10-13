from spym.generic import Spm_image, load_from_gsf
import matplotlib.pyplot as plt
import numpy as np

file_name = 'wg_berk-0deg_n2_6um_000_TR.gsf'
#file_name = 'BMG8-zr60cu30al10_375c_10mN_6um_000_TR.gsf'
image =load_from_gsf(file_name)
image.change_xy_unit('um')
image.change_z_unit('nm')
X, Y, Z = image.get_xyz()
xlabel, ylabel, zlabel = image.get_labels()
# Let's change the center the coordinates on the indent
loc = image.get_min_position()
xc, yc = X[loc], Y[loc] 
image.set_center(xc = xc, yc = yc)
X1, Y1, Z1 = image.get_xyz()
fig = plt.figure()
ax = fig.add_subplot(121)
ax.set_aspect('equal')
plt.contourf(X,Y,Z)
plt.grid()
plt.xlabel(xlabel)
plt.ylabel(ylabel)
ax = fig.add_subplot(122)
ax.set_aspect('equal')
plt.contourf(X1,Y1,Z1)
plt.grid()
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.show()
