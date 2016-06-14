from spym.generic import Spm_image, load_from_gsf
from spym.indentation import Watershed
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np



file_name = 'planilux_berk_10mN_6um.gsf'
# This script is intended to be used on "Gwyddion Simple File" (GSF) files. This choice is motivated by the fact that Gwyddion can open most of the proprietary files standards produced by AFM/SPM commercial devices and can then save them to the open format GSF.
name = 'planilux_berk_10mN_6um.gsf' # GSF File path


# GRAPHICAL SETTINGS
levels = 100 # Number of levels on the color maps
fs = 12.     # Font size
alpha = 2.5  # Cross section tilt 
med_filt_size = None # If the image is noisy, a median filter can be applied, put a number greater than 1 (eg. 3).
cmap = cm.jet

# IMAGE PREPROCESSING
image = load_from_gsf(file_name) # Image loading
image = image.crop(cx0 = 10, cx1= 10, cy0 = 10, cy1= 10) # Light crop in order to remove artifacts on the extremities of the scans
image.change_xy_unit('um') # Setting xy units
image.change_z_unit('nm')  # Setting z unit
xlabel, ylabel, zlabel = image.get_labels() # Getting labels for ploting purpose
X0, Y0, Z0 = image.get_xyz()  # Getting raw image data
image.center_min()            # Offseting to center the bottom of the indent
X1, Y1, Z1 = image.get_xyz()  # Getting centered image data for masking purpose
R2 = X1**2 + Y1**2            # Building a mask 
r = min( X1.max(), -X1.min(),  Y1.max(), -Y1.min())
k = .8
mask = np.where(R2 > r**2 * k**2, 1., np.nan )           # Circular mask filled with 1 and NaN
image, trash = image.line_fit(mask = mask)
image = image.crop(cx0 = 40, cx1= 40, cy0 = 40, cy1= 40) # Cropping to keep only the indent
X, Y, Z = image.get_xyz()     # Final image data
contour, area = Watershed(image, angle = 2.5)

# Plotting
fig = plt.figure(0)
plt.clf()
ax = fig.add_subplot(221)
ax.set_aspect('equal')
plt.title('Raw image')
grad = plt.contourf(X0, Y0, Z0, levels, cmap = cmap)
plt.grid(linewidth = 1.)
#plt.xlabel(xlabel, fontsize = fs)
plt.ylabel(ylabel, fontsize = fs)
cbar = plt.colorbar(grad)
#cbar.set_label(zlabel, fontsize = fs)

ax = fig.add_subplot(222)
ax.set_aspect('equal')
plt.title('Masked image')
grad = plt.contourf(X1, Y1, Z1 * mask, levels, cmap = cmap)
plt.grid(linewidth = 1.)
#plt.xlabel(xlabel, fontsize = fs)
#plt.ylabel(ylabel, fontsize = fs)
cbar = plt.colorbar(grad)
cbar.set_label(zlabel, fontsize = fs)


ax = fig.add_subplot(223)
ax.set_aspect('equal')
plt.title('Corrected image')
grad = plt.contourf(X, Y, Z, levels, cmap = cmap)
plt.grid(linewidth = 1.)
plt.xlabel(xlabel, fontsize = fs)
plt.ylabel(ylabel, fontsize = fs)
cbar = plt.colorbar(grad)
#cbar.set_label(zlabel, fontsize = fs)

ax = fig.add_subplot(224)
ax.set_aspect('equal')
plt.title('Contact Area: $A_c = {0:.2f}$ {1}$^2$'.format(area, xlabel))
grad = plt.contourf(X, Y, Z, levels, cmap = cmap)
plt.grid(linewidth = 1.)
plt.xlabel(xlabel, fontsize = fs)
#plt.ylabel(ylabel, fontsize = fs)
cbar = plt.colorbar(grad)
cbar.set_label(zlabel, fontsize = fs)
plt.plot(contour[0], contour[1], 'k-', linewidth = 2.)


plt.show()
