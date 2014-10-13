from spym.generic import Spm_image, load_from_gsf
import matplotlib.pyplot as plt
import numpy as np

#file_name = 'wg_berk-0deg_n2_6um_000_TR.gsf'
file_name = 'BMG8-zr60cu30al10_375c_10mN_6um_000_TR.sf'
file_name = 'BMG6-Zr60Cu30Al10-asQ_003_TR.gsf'
image =load_from_gsf(file_name)
image.change_xy_unit('um')
image.change_z_unit('nm')
image = image.crop(5,5,5,5) # Removing bad pixels on the perimeter of the image
image.center_min()
X, Y, Z = image.get_xyz()
levels = 20
xlabel, ylabel, zlabel = image.get_labels()
# Let's try it with no mask
image2, residual2 = image.plane_fit()
X2, Y2, Z2 = image2.get_xyz()
Xr2, Yr2, Zr2 = residual2.get_xyz()
# Here we want to remove a circle around the indent
r = 2 # Radius of the circle in um
R2 = X**2 + Y**2
mask = np.where(R2 > r**2, Z, np.nan )
image3, residual3 = image.plane_fit(mask = mask)
X3, Y3, Z3 = image3.get_xyz()
Xr3, Yr3, Zr3 = residual3.get_xyz()
fig = plt.figure()
plt.clf()


# Original image
ax = fig.add_subplot(331)
ax.set_aspect('equal')
grad = plt.contourf(X, Y, Z, levels)
cbar = plt.colorbar(grad)
plt.xticks([])
plt.yticks([])
cbar.set_label(zlabel)
plt.title('Original image')
# Plane fit without mask
ax = fig.add_subplot(334)
ax.set_aspect('equal')
grad = plt.contourf(X, Y, Z, levels)
cbar = plt.colorbar(grad)
plt.xticks([])
plt.yticks([])
plt.title('No mask')
ax = fig.add_subplot(335)
ax.set_aspect('equal')
grad = plt.contourf(X2, Y2, Z2, levels)
cbar = plt.colorbar(grad)
plt.xticks([])
plt.yticks([])
plt.title('Plane fit wo. mask')
cbar.set_label(zlabel)
ax = fig.add_subplot(336)
ax.set_aspect('equal')
grad = plt.contourf(Xr2, Yr2, Zr2, levels)
cbar = plt.colorbar(grad)
plt.xticks([])
plt.yticks([])
cbar.set_label(zlabel)
plt.title('Residual plane')
cbar.set_label(zlabel)
# Plane fit with mask
ax = fig.add_subplot(337)
ax.set_aspect('equal')
grad = plt.contourf(X, Y, mask, levels)
cbar = plt.colorbar(grad)
plt.xticks([])
plt.yticks([])
plt.title('Masked image')
cbar.set_label(zlabel)
ax = fig.add_subplot(338)
ax.set_aspect('equal')
grad = plt.contourf(X3, Y3, Z3, levels)
plt.xticks([])
plt.yticks([])
cbar = plt.colorbar(grad)
cbar.set_label(zlabel)
plt.title('Plane fit w. mask')
cbar.set_label(zlabel)
ax = fig.add_subplot(339)
ax.set_aspect('equal')
grad = plt.contourf(Xr3, Yr3, Zr3, levels)
cbar = plt.colorbar(grad)
plt.xticks([])
plt.yticks([])
cbar.set_label(zlabel)
plt.title('Residual plane')
cbar.set_label(zlabel)
plt.show()

