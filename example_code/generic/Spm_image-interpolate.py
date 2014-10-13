from spym.generic import Spm_image, load_from_gsf
import matplotlib.pyplot as plt
import numpy as np

file_name = 'wg_berk-0deg_n2_6um_000_TR.gsf'
#file_name = 'BMG8-zr60cu30al10_375c_10mN_6um_000_TR.gsf'
image =load_from_gsf(file_name)
image.change_xy_unit('um')
image.change_z_unit('nm')
image = image.crop(5,5,5,5)
image.center_min()
X, Y, Z = image.get_xyz()
r = 2.
mask = np.where(X[0]**2 > r**2, 0., np.nan )
image, residual = image.line_fit(mask = mask)
X, Y, Z = image.get_xyz()
xlabel, ylabel, zlabel = image.get_labels()
# Let's define some paths:
s = np.linspace(-5., 5., 100)
# A tilted line to produce a section
angle0 = 55.
xc = 0.
yc = 0.
x0 = np.cos(np.radians(angle0)) * s + xc
y0 = np.sin(np.radians(angle0)) * s + yc
X0, Y0, Z0 = image.interpolate(x0, y0)
x1 = np.cos(np.radians(angle0 + 120.)) * s + xc
y1 = np.sin(np.radians(angle0 + 120.)) * s + yc
X1, Y1, Z1 = image.interpolate(x1, y1)
x2 = np.cos(np.radians(angle0 - 120.)) * s + xc
y2 = np.sin(np.radians(angle0 - 120.)) * s + yc
X2, Y2, Z2 = image.interpolate(x2, y2)
levels = 100
fig = plt.figure(0)
plt.clf()
ax = fig.add_subplot(121)
ax.set_aspect('equal')
plt.title('Original + Plane Fit')
plt.grid()
grad = plt.contourf(X,Y,Z, levels)
plt.plot(X0, Y0, 'r-', linewidth = 2.)
plt.plot(X1, Y1, 'b-', linewidth = 2.)
plt.plot(X2, Y2, 'g-', linewidth = 2.)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
ax = fig.add_subplot(122)
plt.title('Sections')
plt.plot(s, Z0, 'r-', linewidth = 2.)
plt.plot(s, Z1, 'b-', linewidth = 2.)
plt.plot(s, Z2, 'g-', linewidth = 2.)
plt.xlabel(xlabel)
plt.ylabel(zlabel)
plt.show()
