from spym.generic import Spm_image, load_from_gsf
import matplotlib.pyplot as plt
import numpy as np

file_name = 'wg_berk-0deg_n2_6um_000_TR.gsf'
#file_name = 'BMG8-zr60cu30al10_375c_10mN_6um_000_TR.gsf'
image = load_from_gsf(file_name)
image.change_z_unit('nm')
image.change_xy_unit('um')
X, Y, Z = image.get_xyz()
loc = image.get_min_position()
xc, yc = X[loc], Y[loc] 
image.set_center(xc = xc, yc = yc)
X, Y, Z = image.get_xyz()
xlabel, ylabel, zlabel = image.get_labels()
# Let's crop
image1 = image.crop(cx0 = -100, cx1 = -50, cy0 = 20, cy1=80)
X1, Y1, Z1 = image1.get_xyz()
levels = 100
fig = plt.figure(0)
plt.clf()
ax = fig.add_subplot(121)
ax.set_aspect('equal')
plt.title('Original')
plt.grid()
grad = plt.contourf(X,Y,Z, levels)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
ax = fig.add_subplot(122)
ax.set_aspect('equal')
plt.title('Cropped')
plt.grid()
grad = plt.contourf(X1,Y1,Z1, levels)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.show()
