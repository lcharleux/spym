from spym.generic import Spm_image
from spym.indentation import Indenter_orientation
import matplotlib.pyplot as plt
import numpy as np


#file_name = 'wg_berk-0deg_n2_6um_000_TR.txt'
#file_name = 'BMG8-zr60cu30al10_375c_10mN_6um_000_TR.txt'
file_name = 'BMG6-Zr60Cu30Al10-asQ_003_TR.txt'
image = Spm_image()
image.load_from_file(file_name)
image = image.crop(cx0 = 10, cx1= 10, cy0 = 10, cy1= 10)
image.center_min()
X, Y, Z = image.get_xyz()
edges = 3
R, T, Z = image.get_rtz(rmax = 'min')
rotation, spectrum, edges, rtz = Indenter_orientation(image, edges)
t, z = spectrum # Spectrum 
te, ze = edges  # Detected edges
Rl, Tl, Zl = rtz # Cylindrical interpolation of the laplacian of the image
n_levels = 100
fig = plt.figure(0)
ax = fig.add_subplot(221, polar = True)
plt.title('Original image')
plt.contourf(T, R, Z, n_levels)
plt.yticks([])
plt.xticks(np.radians([0., 45., 135., 180., 225., 315.]))
ax = fig.add_subplot(222, polar = True)
plt.title('Laplacian')
plt.contourf(Tl, Rl, Zl, n_levels)
plt.yticks([])
plt.xticks(np.radians([0., 45., 135., 180., 225., 315.]))
ax = fig.add_subplot(223)
plt.title('Spectrum')
plt.plot(t,z, 'b-', label = 'spectrum')
plt.plot(te, ze, 'or', label = 'peaks')
plt.legend(loc = 'lower center', ncol = 2)
plt.xlabel(r'Azimut')
plt.ylabel(r'Amplitude')
plt.xticks(te, ['{0:.1f}'.format(tte) + r'$^O$' for tte in te])
plt.grid()
plt.yticks([])
ax = fig.add_subplot(224, polar = True)
plt.title('Detected edges')
plt.contourf(Tl, Rl, Zl, n_levels)
plt.yticks([])
plt.xticks(np.radians(te))
plt.show()




