from spym.generic import Spm_image, load_from_gsf
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

#file_name = 'wg_berk-0deg_n2_6um_000_TR.gsf'
file_name = 'BMG8-zr60cu30al10_375c_10mN_6um_000_TR.gsf'
image = load_from_gsf(file_name) # Load the image
image.change_xy_unit('um')
image.change_z_unit('nm')
xlabel, ylabel, zlabel = image.get_labels()
X, Y, Z = image.get_xyz()
loc = image.get_min_position()
xc, yc = X[loc], Y[loc] 
image.set_center(xc = xc, yc = yc)
X, Y, Z = image.get_xyz()
image = image.rotate(angle = 30., tx = 0., ty = 0.)
levels = 10
Xr, Yr, Zr = image.get_xyz()
fig = plt.figure(0)
plt.clf()
ax = fig.add_subplot(121)
ax.set_aspect('equal')
plt.title('Original')
grad = plt.contourf(X,Y,Z, levels)
plt.contour(X,Y,Z, levels, colors = 'black')
cbar = plt.colorbar(grad)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
cbar.set_label(zlabel)
plt.grid()
ax = fig.add_subplot(122)
ax.set_aspect('equal')
plt.title('Rotated')
grad = plt.contourf(Xr,Yr,Zr, levels)
plt.contour(Xr,Yr,Zr, levels, colors = 'black')
cbar = plt.colorbar(grad)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
cbar.set_label(zlabel)
plt.grid()
plt.show()
