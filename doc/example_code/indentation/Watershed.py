from spym.generic import Spm_image
from spym.indentation import Watershed
import matplotlib.pyplot as plt
import numpy as np


#file_name = 'wg_berk-0deg_n2_6um_000_TR.txt'
#file_name = 'bmg14_10mN_4um_001_TF.txt'
#file_name = 'BMG8-zr60cu30al10_375c_10mN_6um_000_TR.txt'
#file_name = 'BMG6-Zr60Cu30Al10-asQ_003_TR.txt'
file_name = 'fq_hys_10mN_4um_003_TF.txt'
image = Spm_image()
image.load_from_file(file_name)
image = image.crop(cx0 = 10, cx1= 10, cy0 = 10, cy1= 10)
image.change_xy_unit('m')
image.change_z_unit('m')
image.center_min()
X0, Y0, Z0 = image.get_xyz()
R2 = X0**2 + Y0**2
r = min( X0.max(), -X0.min(),  Y0.max(), -Y0.min())
k = .8
mask = np.where(R2 > r**2 * k**2, 1., np.nan )
image.paraboloid_fit(mask = mask)
image, trash = image.line_fit(mask = mask)
X, Y, Z = image.get_xyz()
xy, area = Watershed(image, angle = 2., size = 1)
x, y = xy[0], xy[1]
levels = 100
fig = plt.figure(0)
plt.clf()
plt.gca().set_aspect('equal')
plt.contourf(X, Y, Z, levels)
plt.colorbar()
plt.plot(x, y, 'k-')

plt.show()
