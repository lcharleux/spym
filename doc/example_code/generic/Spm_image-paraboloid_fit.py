from spym.generic import Spm_image, load_from_gsf
import matplotlib.pyplot as plt
import numpy as np

#file_name = 'wg_berk-0deg_n2_6um_000_TR.gsf'
file_name = 'WG-air_10mN_000_TF.gsf'
#file_name = 'BMG8-zr60cu30al10_375c_10mN_6um_000_TR.gsf'
image = load_from_gsf(file_name) # Load the image
image.change_xy_unit('um')
image.change_z_unit('nm')
X, Y, Z = image.get_xyz()
levels = 20
xlabel, ylabel, zlabel = image.get_labels()
X0, Y0, Z0   = image.get_xyz()
image1, trash = image.plane_fit()
X1, Y1, Z1   = image1.get_xyz()
image2, trash = image.paraboloid_fit()
X2, Y2, Z2   = image2.get_xyz()
image3, trash = image.line_fit()
X3, Y3, Z3   = image3.get_xyz()
image4, trash = image3.paraboloid_fit()
X4, Y4, Z4   = image4.get_xyz()
fig = plt.figure()
plt.clf()


# Original image
ax = fig.add_subplot(321)
ax.set_aspect('equal')
grad = plt.contourf(X0, Y0, Z0, levels)
cbar = plt.colorbar(grad)
plt.xticks([])
plt.yticks([])
cbar.set_label(zlabel)
plt.title('Original image')
# Plane fit without mask
ax = fig.add_subplot(322)
ax.set_aspect('equal')
grad = plt.contourf(X1, Y1, Z1, levels)
cbar = plt.colorbar(grad)
cbar.set_label(zlabel)
plt.xticks([])
plt.yticks([])
plt.title('Plane')
ax = fig.add_subplot(323)
ax.set_aspect('equal')
grad = plt.contourf(X2, Y2, Z2, levels)
cbar = plt.colorbar(grad)
plt.xticks([])
plt.yticks([])
plt.title('Paraboloid')
cbar.set_label(zlabel)
ax = fig.add_subplot(324)
ax.set_aspect('equal')
grad = plt.contourf(X3, Y3, Z3, levels)
cbar = plt.colorbar(grad)
plt.xticks([])
plt.yticks([])
plt.title('Line')
cbar.set_label(zlabel)
ax = fig.add_subplot(325)
ax.set_aspect('equal')
grad = plt.contourf(X4, Y4, Z4, levels)
cbar = plt.colorbar(grad)
plt.xticks([])
plt.yticks([])
plt.title('Line + Paraboloid')
cbar.set_label(zlabel)
plt.show()

