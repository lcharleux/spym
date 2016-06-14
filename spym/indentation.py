# -*- coding: utf-8 -*-

import numpy as np
import scipy.ndimage as nd
from scipy import signal
import copy
import matplotlib.pyplot as plt

def Indenter_orientation(image, edges, rmax = 'min', width = .5):
  '''
  Finds the orientation of a pyramidal indenter using the imprint.
  
  :param image: imprint
  :type image: ``spy.generic.Spm_image`` instance
  :param edges: number of edges on the indenter.
  :type edges: int
  :param rmax: max radius for the polar interpolation. 
  :type rmax: float, 'min' or 'max'
  
  
  .. plot:: example_code/indentation/Indenter_orientation.py
    :include-source: 
  '''
  image = copy.deepcopy(image)
  image.center_min()
  image.data = nd.filters.laplace(image.data)
  R, T, Z = image.get_rtz(rmax = rmax)
  z = Z.sum(axis = 1)
  t = np.degrees(T[:,0])
  ze, te = [], []
  nw = int(1. / float(edges) * width / 2. * float(len(z)))
  zc, tc = z.copy(), t.copy() 
  l = len(zc)
  for edge in xrange(edges):
    zm = zc.max()
    zm_loc = np.where(zc == zm)[0][0]
    to_nullify = np.arange(2 * nw + 1) - nw + zm_loc
    to_nullify = np.where(to_nullify > l - 1, to_nullify - l, to_nullify)
    ze.append(zm)
    te.append(t[zm_loc])
    zc[to_nullify] = 0.
  rotation = 0.  
  te, ze = zip(*sorted(zip(te, ze)))
  for i in xrange(len(te)): rotation += te [i] - 360. / edges * i        
  rotation = rotation / float(edges)
  return rotation, (t,z), (te, ze), (R, T, Z)
  
def Pyramidal_indenter(image, sides = 3, angle = 65.27, rotation = 0. ):
  '''
  Builts a pyramidal indenter based on an existing image. The image is used to find xy and z units, lx and ly and the position of the origin. The tip of the indenter is always at X = Y = 0.  
  
  :param image: the image to use.
  :type image: ``spym.generic.Spm_image`` instance.
  :param sides: number of sides.
  :type sides: int
  :param angle: angle between the axis and the faces. 
  :type angle: float
  :param rotation: rotation or the sides in the (x,y) plane in degrees.
  :type rotation: float
  
  .. plot:: example_code/indentation/Pyramidal_indenter.py
    :include-source:  
  '''
  out = image.copy()
  X,Y,Z = image.get_xyz()
  theta = np.linspace(0., 360., sides, endpoint = False) + 360. / float(sides) / 2. + rotation
  Faces = []
  for t in theta : 
     D = X * np.cos(np.radians(t)) + Y * np.sin(np.radians(t))
     F = D / np.tan(np.radians(angle))
     Faces.append(F)
  Faces = np.array(Faces)
  Z = np.amax(Faces, axis = 0)
  Z = Z * out.get_xy_factor() / out.get_z_factor()
  out.data = Z
  
  return out

def Conical_indenter(image, angle = 70.27):
  '''
  Builts a conical indenter based on an existing image. The image is used to find xy and z units, lx and ly and the position of the origin. The tip of the indenter is always at X = Y = 0.  
  
  :param image: the image to use.
  :type image: ``spym.generic.Spm_image`` instance.
   
  '''
  out = image.copy()
  X,Y,Z = image.get_xyz()
  R = (X**2 + Y**2)**.5 
  Z = R / np.tan(np.radians(angle)) 
  Z = Z * out.get_xy_factor() / out.get_z_factor()
  out.data = Z
  return out

def Contact_area(image, indenter):
  '''
  Returns the contact area between an image, an indenter and at a given depth.
  
  :param image: the image to use.
  :type image: ``spym.generic.Spm_image`` instance.
  :param indenter: the image to use.
  :type indenter: ``spym.generic.Spm_image`` instance.
  :rtype: float
  
  
  
  
  .. plot:: example_code/indentation/Contact_area.py
    :include-source:  
  '''  
  X, Y, Z = image.get_xyz()
  Xi, Yi, Zi = indenter.get_xyz()
  Zeq = Zi - Z
  A = (Zeq <= 0. ).sum() * image.pixel_area()
  fig = plt.figure('trash')
  plt.clf()
  levels = [-1, 0., 1.]
  cont = plt.contour(Xi, Yi, Zeq, levels)
  path = cont.collections[1].get_paths()[0].vertices.copy()
  path = path.transpose()
  x, y = copy.copy(path[0]), copy.copy(path[1])
  plt.close()
  return (x, y), A
  #return 'trash', A
  
def Watershed(image, angle = 2.5, size = None, filter_type = 'med'):
  '''
  Returns the contour of the contact area according to the method introduced in [1]_ . 
  
  .. [1] L. Charleux, V. Keryvin, M. Nivard, J.-P. Guin, J.-C. Sangleboeuf, and Y. Yokoyama, "A method for measuring the contact area in instrumented indentation testing by tip scanning probe microscopy imaging", Acta Mater., vol. 70, pp. 249-258  `DOI <http://dx.doi.org/10.1016/j.actamat.2014.02.036>`_. 
  
  :param image: the image to use.
  :type image: ``spym.generic.Spm_image`` instance.
  :param angle: cross section tilt angle 
  :type angle: float
  :param size: filter core size, if None, no filter is applied.
  :type size: int or float
  :param filter_type: "med" for floating average filter and "gauss" for gaussian filter.
  :type filter_type: string
  
  .. plot:: example_code/indentation/Watershed.py
    :include-source:  
  
  '''
  image = image.copy()
  xy_factor = image.get_xy_factor()
  z_factor = image.get_z_factor()
  image.change_xy_unit('m')
  image.change_z_unit('m')
  if size != None:
    if filter_type == 'med':
      image.data = signal.medfilt(image.data, size)
    if filter_type == 'gauss':
      image.data = nd.gaussian_filter(image.data, size)
  R, T, Z = image.get_rtz()
  rl, tl, zl = [], [], []
  Zs = Z.copy()
  Zs += -np.tan(np.radians(angle)) * R
  
  for i in xrange(len(Z)):
    zs = Zs[i]
    z = Z[i]
    r = R[i]
    t = T[i]
    zm = np.nanmax(zs)
    loc = np.where(zs == zm)[0][0]
    zl.append(z[loc])
    rl.append(r[loc])
    tl.append(t[loc])
  rl = np.array(rl) / xy_factor
  tl = np.array(tl)
  zl = np.array(zl) / z_factor
  Zs = Zs / z_factor
  x = rl * np.cos(tl)
  y = rl * np.sin(tl)
  area = 0.
  for i in xrange(len(x)-1):
    area += (x[i]*y[i+1]-x[i+1]*y[i])/2.
  
  return (x, y, rl, tl, zl), area
    

