import numpy as np
from scipy import optimize, interpolate, ndimage
import copy
from StringIO import StringIO   # StringIO behaves like a file object

def Measure(spm, N):
  '''
  Measures a scratch oriented along y axis. Treated images have to be plane or line fitted before using this function to have accurate results. The method returns a dict containing the xyz trajectory of the bottom line, the left pile-up and the right pile-up. It also contains average measurement:
  
  * Depth of the groove.
  * Heigth of the pile up (left, rigth and average)
  * Width of the groove.  
  
  :param N: number of subdivision of the image along y axis. N must respect 1 <=  N <= self.y_samples()
  :type N: integer
  :rtype: dict 
  
  .. plot:: example_code/scratch/Measure.py
      :include-source:    
  '''  
  band_size = spm.ly / N
  X, Y, Z = spm.get_xyz()
  X_bottom,   Y_bottom,   Z_bottom   = [], [], []
  X_topleft,  Y_topleft,  Z_topleft  = [], [], []
  X_topright, Y_topright, Z_topright = [], [], []
  for i in xrange(N):
    loc = np.where(np.logical_and( Y >= i * band_size, Y <=  (i+1) * band_size  ) )
    x = X[loc]
    y = Y[loc]
    z = Z[loc]
    z_bottom = z.min()
    x_bottom = x[np.where(z == z_bottom)][0]
    y_bottom = y[np.where(z == z_bottom)][0]
    zl, yl, xl = z[np.where(x < x_bottom)], y[np.where(x < x_bottom)], x[np.where(x < x_bottom)]
    z_topleft = zl.max()
    x_topleft = xl[np.where(zl == z_topleft)][0]
    y_topleft = yl[np.where(zl == z_topleft)][0]
    zr, yr, xr = z[np.where(x > x_bottom)], y[np.where(x > x_bottom)], x[np.where(x > x_bottom)]
    z_topright = zr.max()
    x_topright = xr[np.where(zr == z_topright)][0]
    y_topright = yr[np.where(zr == z_topright)][0]
    X_bottom.append(x_bottom)
    Y_bottom.append(y_bottom)
    Z_bottom.append(z_bottom)
    X_topleft.append(x_topleft)
    Y_topleft.append(y_topleft)
    Z_topleft.append(z_topleft)
    X_topright.append(x_topright)
    Y_topright.append(y_topright)
    Z_topright.append(z_topright)
  out = {}
  out['xb'] = np.array(X_bottom)
  out['yb'] = np.array(Y_bottom)
  out['zb'] = np.array(Z_bottom)
  out['xl'] = np.array(X_topleft)
  out['yl'] = np.array(Y_topleft)
  out['zl'] = np.array(Z_topleft)
  out['xr'] = np.array(X_topright)
  out['yr'] = np.array(Y_topright)
  out['zr'] = np.array(Z_topright)
  out['depth'] = -out['zb'].mean()
  out['r_heigth'] = out['zr'].mean()
  out['l_heigth'] = out['zl'].mean()
  out['heigth']   = (out['r_heigth'] + out['l_heigth']) / 2.
  out['width']    = (out['xr'] - out['xl']).mean()
  return out 
 
  
    
    
