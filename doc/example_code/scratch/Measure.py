from spym.generic import Spm_image
from spym.scratch import Measure
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

file_name = 'bmg04_002_001_TF.dat'
image = Spm_image()
image.load_from_file(file_name)
image.change_z_unit('um')
X, Y, Z = image.get_xyz()
x, z = X[0], Z[0]
mask = np.where(np.logical_and(x>2., x<8.), np.nan, 0.)
image, trash = image.line_fit(mask = mask)
X, Y, Z = image.get_xyz()
xlabel, ylabel, zlabel = image.get_labels()
m = Measure(image, 64)
lw = 2.
title = 'Width = ${0:1.2f}${1}, Depth = ${2:1.2f}${3} , Pile-up = ${4:1.2f}${3}'
fig = plt.figure()
plt.clf()
grad = plt.contourf(X,Y,Z, 100, cmap = cm.gray)
cbar = plt.colorbar(grad)
cbar.set_label(zlabel)
plt.plot(m['xl'], m['yl'], label = 'Left', linewidth = lw)
plt.plot(m['xb'], m['yb'], label = 'Bottom', linewidth = lw)
plt.plot(m['xr'], m['yr'], label = 'Right', linewidth = lw)
plt.legend()
plt.title(title.format(m['width'], xlabel, m['depth'], zlabel, m['heigth'] ))
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.show()
