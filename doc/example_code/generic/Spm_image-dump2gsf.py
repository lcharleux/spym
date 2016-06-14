from spym.generic import load_from_gsf
import numpy as np


file_name = 'WG-air_10mN_001_TF.gsf'
file_name = 'BMG8-zr60cu30al10_375c_10mN_6um_000_TR.gsf'
image = load_from_gsf(file_name) # Load the image
# Perform some operations
image.change_xy_unit('um')
image.change_z_unit('nm')
image = image.crop(5,5,5,5)
X, Y, Z = image.get_xyz()
xi = X[image.get_min_position()]
yi = Y[image.get_min_position()]
r = 2.
mask = np.where((X-xi)**2 + (Y-yi)**2 > r**2, 0., np.nan )
image, residual = image.line_fit(mask = mask)

image.center_min()
# Save it again
image.dump2gsf(file_name.replace('.', '_linefit.'))
