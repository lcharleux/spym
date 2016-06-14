from spym.generic import Spm_image, load_from_gsf
import matplotlib.pyplot as plt
import numpy as np
from spym.indentation import Indenter_orientation

file_name = 'wg_berk-0deg_n2_6um_000_TR.gsf'
#file_name = 'BMG8-zr60cu30al10_375c_10mN_6um_000_TR.gsf'
image =load_from_gsf(file_name)
image.change_z_unit('nm')
image.change_xy_unit('um')
image = image.crop(cx0 = 5, cx1= 5, cy0 = 5, cy1= 5)
image, trash = image.plane_fit()
image.center_min()
X0, Y0, Z0 = image.get_xyz()
R2 = X0**2 + Y0**2
r = min( X0.max(), -X0.min(),  Y0.max(), -Y0.min())
k = .8
mask = np.where(R2 > r**2 * k**2, 1., np.nan )
image, trash = image.line_fit(mask = mask)
rotation, spectrum, edges, rtz = Indenter_orientation(image, 3)
angles, peaks = edges  # Detected 1edges
X, Y, Z = image.get_xyz()
zmax, zmin = Z.max(), -.5 * Z.max()
levels = np.linspace(zmin, zmax, 20)


fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.set_aspect("equal")
plt.grid()
grad = plt.contourf(X, Y, Z, levels)
plt.contour(X, Y, Z, levels, colors = 'black', linewidths = 0.2)
cbar = plt.colorbar(grad)
cbar.set_label("Altitude, $z$ [nm]")
#plt.clim(-20., 20.)
ax2 = fig.add_subplot(1,2,2)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
plt.grid()

for angle in angles:
 x, y, s, z = image.section(angle = angle)
 ax1.plot(x, y)
 ax2.plot(s, z)
plt.show()
