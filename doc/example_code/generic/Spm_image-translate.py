from spym.generic import Spm_image, load_from_gsf
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

#file_name = 'wg_berk-0deg_n2_6um_000_TR.gsf'
file_name = 'BMG8-zr60cu30al10_375c_10mN_6um_000_TR.gsf'
image =load_from_gsf(file_name)
image.change_xy_unit('um')
image.change_z_unit('nm')
X, Y, Z = image.get_xyz()
xlabel, ylabel, zlabel = image.get_labels()
# Translation
image = image.translate(nx = 30, ny = -50)
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
ax = fig.add_subplot(122)
ax.set_aspect('equal')
plt.title('Translate')
grad = plt.contourf(Xr,Yr,Zr, levels)
plt.contour(Xr,Yr,Zr, levels, colors = 'black')
cbar = plt.colorbar(grad)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
cbar.set_label(zlabel)
plt.show()
