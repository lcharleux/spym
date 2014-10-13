from spym.generic import Spm_image
from spym.indentation import Indenter_orientation, Pyramidal_indenter, Contact_area
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

#file_name = 'fq_hys_10mN_4um_003_TF.txt'
#file_name = 'wg_berk-0deg_n2_6um_000_TR.txt'
#file_name = 'BMG8-zr60cu30al10_375c_10mN_6um_000_TR.txt'
file_name = 'BMG6-Zr60Cu30Al10-asQ_003_TR.txt'
image = Spm_image()
image.load_from_file(file_name)
image.change_z_unit('nm')
xlabel, ylabel, zlabel = image.get_labels()
image = image.crop(cx0 = 5, cx1= 5, cy0 = 5, cy1= 5)
image.center_min()
X0, Y0, Z0 = image.get_xyz()
R2 = X0**2 + Y0**2
r = min( X0.max(), -X0.min(),  Y0.max(), -Y0.min())
k = .9
mask = np.where(R2 > r**2 * k**2, 1., np.nan )
#image, trash = image.plane_fit(mask = mask)
image, trash = image.line_fit(mask = mask)
#image = image.crop(cx0 = 50, cx1= 50, cy0 = 50, cy1= 50)
X, Y, Z = image.get_xyz()
rotation, spectrum, edges, rtz = Indenter_orientation(image, 3)
indenter = Pyramidal_indenter(image, rotation = rotation)

dh = 14.    # Berkovic tip truncated length.
hmax = 280. # Maximum disp from the load disp curve for BMG6.
hu = Z.min() + hmax
indenter.data += Z.min() - dh - hu / 2.  
Xi, Yi, Zi = indenter.get_xyz()
Zeq = Zi - Z
(x, y), Ac = Contact_area(image, indenter)
A_in = np.where(Zeq <= 0., Zeq, np.nan) 
A_out = np.where(Zeq > 0., Zeq, np.nan)

levels = 100
dh = 14.
fig = plt.figure(0)
ax = fig.add_subplot(321)
plt.title('Raw image')
ax.set_aspect('equal')
plt.grid()
plt.ylabel(ylabel)
plt.contourf(X0, Y0, Z0, levels)
cbar = plt.colorbar()
ax.set_xticklabels([])
ax = fig.add_subplot(322)
plt.title('Masked image')
ax.set_aspect('equal')
plt.grid()
plt.contourf(X0, Y0, Z0 * mask, levels)
cbar = plt.colorbar()
ax.set_xticklabels([])
ax.set_yticklabels([])
cbar.set_label(zlabel)
ax = fig.add_subplot(323)
plt.title('Untilted image')
ax.set_aspect('equal')
plt.grid()
plt.ylabel(ylabel)
plt.contourf(X, Y, Z, levels)
plt.colorbar()
ax.set_xticklabels([])
ax = fig.add_subplot(324)
plt.title('Indenter')
ax.set_aspect('equal')
plt.grid()
plt.contourf(Xi, Yi, Zi, levels)
cbar = plt.colorbar()
ax.set_xticklabels([])
ax.set_yticklabels([])
cbar.set_label(zlabel)
ax = fig.add_subplot(325)
plt.title('Equivalent indenter')
ax.set_aspect('equal')
plt.grid()
plt.xlabel(xlabel)
plt.contourf(Xi, Yi, Zeq, levels)
cbar = plt.colorbar()
ax = fig.add_subplot(326)
plt.title('Ac = {0:1.2e}'.format(Ac))
ax.set_aspect('equal')
plt.grid()
plt.xlabel(xlabel)
grad = plt.contourf(Xi, Yi, A_in,levels, cmap = cm.jet)
cbar = plt.colorbar(grad)
cbar.set_label(zlabel)
plt.contourf(Xi, Yi, A_out,levels, cmap = cm.gray)
plt.plot(x, y)
ax.set_yticklabels([])
plt.show()
