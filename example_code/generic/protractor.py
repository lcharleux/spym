from spym.generic import Spm_image, protractor
import matplotlib.pyplot as plt
import numpy as np

file_name = 'wg_berk-0deg_n2_6um_000_TR.txt'
image = Spm_image()
image.load_from_file(file_name)
# Here we want to remove a band around the indent
r = 2 # Radius of the band in um
X, Y, Z = image.get_xyz()
xi = X[image.get_min_position()]
yi = Y[image.get_min_position()]
mask = np.where((X[0]-xi)**2 > r**2, 0., np.nan )
image, residual = image.line_fit(mask = mask)
image.change_z_unit('um')
X, Y, Z = image.get_xyz()
R = ((X-xi)**2 + (Y-yi)**2)**.5
mask = np.where(R < 1.5, 0., np.nan )
xlabel, ylabel, zlabel = image.get_labels()
levels = 100
xc = X[image.get_min_position()][0]
yc = Y[image.get_min_position()][0]
fig = plt.figure(0)
ax = fig.add_subplot(111)
ax.set_aspect('equal')
grad = plt.contourf(X,Y,Z+ mask, levels)
plt.xticks([])
plt.yticks([])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
protractor(ax, 
  nGrad =90,
  r0 = image.lx/8., 
  r1 = image.lx/2.8, 
  xc = xc, 
  yc = yc,  
  fs = 8., 
  lw = 1.)
plt.show()
