from spym.generic import load_from_gsf
import matplotlib.pyplot as plt

file_name = 'WG-air_10mN_001_TF.gsf'
im = load_from_gsf(file_name)
im.change_xy_unit('um')
im.change_z_unit('nm')

plt.figure()
plt.clf()
X,Y,Z = im.get_xyz()
xlabel, ylabel, zlabel = im.get_labels()
plt.xlabel(xlabel)
plt.ylabel(ylabel)
grad = plt.contourf(X,Y,Z, 100)
cbar = plt.colorbar(grad)
cbar.set_label(zlabel)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.gca().set_aspect('equal')
plt.grid()
plt.show()    
