from spym.generic import Spm_image, load_from_gsf
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd
file_name = 'wg_berk-0deg_n2_6um_000_TR.gsf'
#file_name = 'BMG8-zr60cu30al10_375c_10mN_6um_000_TR.gsf'
image =load_from_gsf(file_name)
image.change_xy_unit('um')
image.change_z_unit('nm')
r = 2 # Radius of the band in um
X, Y, Z = image.get_xyz()
xi = X[image.get_min_position()]
mask = np.where((X[0]-xi)**2 > r**2, 0., np.nan )
image, residual = image.line_fit(mask = mask)
X, Y, Z = image.get_xyz()
loc = image.get_min_position()
xc, yc = X[loc], Y[loc] 
image.set_center(xc = xc, yc = yc)
R, T, Z = image.get_rtz(rmax = 'min')
xlabel, ylabel, zlabel = image.get_labels()
fig = plt.figure()
ax = fig.add_subplot(111, polar = True)
grad = plt.contourf(T + np.radians(8.), R, Z, 50)
cbar = plt.colorbar(grad)
plt.yticks([0.5, 1., 1.5]) # Radius
plt.xticks(np.linspace(0., 2. * np.pi, 6, endpoint = False))
plt.ylabel(ylabel)
cbar.set_label(zlabel)
plt.show()
