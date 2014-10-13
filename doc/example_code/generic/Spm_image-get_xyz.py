from spym.generic import Spm_image, load_from_gsf
import matplotlib.pyplot as plt

#file_name = 'wg_berk-0deg_n2_6um_000_TR.gsf'
file_name = 'BMG6-Zr60Cu30Al10-asQ_003_TR.gsf'
image =load_from_gsf(file_name)
image.change_xy_unit('um')
image.change_z_unit('nm')
image = image.crop(5,5,5,5)
image.center_min()
X, Y, Z = image.get_xyz()
xlabel, ylabel, zlabel = image.get_labels()
plt.figure()
grad = plt.contourf(X, Y, Z, 100)
cbar = plt.colorbar(grad)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.grid()
plt.title('Berkovich indent on a BMG at $10 mN$')
cbar.set_label(zlabel)
plt.show()
