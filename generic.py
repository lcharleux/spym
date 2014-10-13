import numpy as np
from scipy import optimize, interpolate, ndimage
import copy
from StringIO import StringIO   # StringIO behaves like a file object
import struct


class Spm_image(object):
  '''
  A class to proceed Scanning Probe Microscopy images using Python.
  '''
  def __init__(self, name = 'SPM image', data = np.array([[0., 1.],[1., 2.]]), lx = 1., ly=1., xc =0., yc = 0., xy_unit = 'm', z_unit = 'm', channel = 'topography'):
    import numpy as np
    self.name = name
    self.channel = channel
    self.lx = lx
    self.ly = ly
    self.xc = xc
    self.yc = yc
    self.xy_unit = xy_unit
    self.z_unit = z_unit
    self.data = np.array(data)
    
  def __repr__(self):
    return '<spym.generic.Spm_image instance: {0}>'.format(self.name)
  
  def __str__(self):
    pattern = '''  * Name = {0}
  * Channel = {1} 
  * x points x y points = {2} x {3}
  * lx x ly = {4} {8} x {5} {8}
  * max / min = {6:.2e} {9} / {7:.2e} {9}'''
    return pattern.format(
      self.name,
      self.channel,
      self.x_samples(), self.y_samples(),
      self.lx, self.ly,
      self.data.min(), self.data.max(), 
      self.xy_unit, self.z_unit)
  
  
  
  def __add__(self, other):
    out = self.copy()
    if isinstance(other, Spm_image):
      other = other.copy()
      other.change_z_unit(self.z_unit)
      other.change_xy_unit(self.xy_unit)
      
    else:
      out.data += other
    return out   
  
  def set_xy_unit(self, xy_unit):
    '''
    Sets x/y unit of the image without modifying its data.
    
    :param xy_unit: x/y unit taken in ['m', 'cm', 'mm', 'um', 'nm']
    :type xy_unit: string
    '''  
    if xy_unit not in ['m', 'cm', 'mm', 'um', 'nm']:
      raise Exception, 'xy_unit must be in ["m", "cm", "mm", "um", "nm"], got {0} instead'.format(xy_unit)
    self.xy_unit = xy_unit
    
    
  def set_z_unit(self, z_unit):
    '''
    Sets z unit of the image without modifying its data.
    
    :param z_unit: z unit taken in ['m', 'cm', 'mm', 'um', 'nm', 'N', 'mN', 'uN', 'nN']
    :type z_unit: string
    '''  
    accepted_units = ['m', 'cm', 'mm', 'um', 'nm', 'N', 'mN', 'uN', 'nN', 'Pa', 'kPa', 'Mpa']
    if z_unit not in accepted_units:
      raise Exception, 'z_unit must be in {0}, got {1} instead'.format(accepted_units, xy_unit)
    self.z_unit = z_unit
  
  def set_lx(self, lx):
    '''
    Sets the length of the image on x axis.
    
    :param lx: length > 0.
    :type lx: float 
    '''
    lx = float(lx)
    if lx <= 0.:
      raise Exception, 'lx must be > 0., got {0} instead'.format(lx)
    self.lx = lx  
  
  def set_ly(self, ly):
    '''
    Sets the length of the image on y axis.
    
    :param ly: length > 0.
    :type ly: float 
    '''
    ly = float(ly)
    if ly <= 0.:
      raise Exception, 'ly must be > 0., got {0} instead'.format(ly)
    self.ly = ly    
  
  def set_center(self, xc = 0., yc = 0.):
    '''
    Sets the center of the image.
        
    :param xc: center of the image along x.
    :type xc: float
    :param yc: center of the image along y.
    :type yc: float 
    
    .. plot:: example_code/generic/Spm_image-set_center.py
      :include-source:    
    '''
    self.xc += xc
    self.yc += yc
  
  def set_name(self, name):
    '''
    Set the name of the image.
    
    :param name: name of the image.
    :type name: string
    '''
    name = str(name)
    self.name = name
  
  def set_data(self, data):
    '''
    Sets data of the image.
    
    :param data: data of the image, must be a rectangular 2D array containing numbers that can be converted to float.
    :type data: ``numpy.array``, ``array.array`` or nested list.
    '''
    data = np.array(data)
    self.data = data
  
  def set_channel(self, channel):
    '''
    Sets the channel description of the image.
    '''
    channel = str(channel)
    self.channel = channel
  
  
    
  
  
  def load_from_file(self, file_name):
    '''
    Loads image attributes from an ASCII file using the :download:`following format <files/wg_berk-30deg_n2_6um_000_TF.dat>` (can be produced by Gwyddion using ascii export):
    
    
    
    :param file_name: name of the file where to read data.
    :type file_name: string
    
    .. plot:: example_code/generic/Spm_image-load_from_file.py
      :include-source:    
    '''
    f = open(file_name, 'r')
    '''
    
    
    header = True
    while header:
      pos = f.tell()
      line = f.readline()
      if line[0] == '#': 
        f.seek(pos)
        header = False
      else:
        words = t.split(':')
        label = words[0].replace('#', '')
        data = words[1]
        if label[0] == ' ': header = header[1:]
        if label[-1] == ' ': header = header[:-1]
        if label == 'Width':  label = 'lx'
        if label == 'Heigth': label = 'ly'
    f = open(file_name, 'r')
    while True:
      pos = f.tell()
      line = f.readline()
      if line[0] == '#': 
        f.seek(pos)
    '''      
    
    channel    = f.readline().split(':')[1].split()[0].lower()
    width_data = f.readline().split(':')[1]
    lx         = float(width_data.split()[0])
    xy_unit    = width_data.split()[1].replace('\xc2\xb5', 'u')
    ly         = float(f.readline().split(':')[1].split()[0])
    z_unit     = f.readline().split(':')[1].split()[0].replace('\xc2\xb5', 'u')
    data       = np.loadtxt(StringIO(f.read()))
    f.close()
    self.set_channel(channel)
    self.set_lx(lx)
    self.set_ly(ly)
    self.set_xy_unit(xy_unit)
    self.set_z_unit(z_unit)
    self.set_data(data)
    self.set_name(file_name)
    
  def get_z_factor(self, unit = None):
    if unit == None: unit = self.z_unit
    if unit in ['MPa'] : z_factor = 1.e6
    if unit in ['kPa'] : z_factor = 1.e3
    if unit in ['m', 'N', 'Pa']: z_factor = 1. 
    if unit in ['mm', 'mN']: z_factor = 1.e-3 
    if unit in ['um', 'uN']: z_factor = 1.e-6
    if unit in ['nm', 'nN']: z_factor = 1.e-9
    return z_factor
    
  def get_xy_factor(self, unit = None):
    if unit == None: unit = self.xy_unit
    if unit in ['m', 'N']: xy_factor = 1. 
    if unit in ['mm', 'mN']: xy_factor = 1.e-3 
    if unit in ['um', 'uN']: xy_factor = 1.e-6
    if unit in ['nm', 'nN']: xy_factor = 1.e-9
    return xy_factor
    
  def x_samples(self):
    '''
    Returns the number of samples along x axis.
    
    :rtype: int
    '''
    return len(self.data[0])
  
  def y_samples(self):
    '''
    Returns the number of samples along y axis.
    
    :rtype: int
    '''
    return len(self.data)
  
  def change_z_unit(self, new_unit):
    '''
    Changes the z unit to a new one and corrects the values in ``self.data``.
    '''
    factor = self.get_z_factor() / self.get_z_factor(new_unit)
    self.set_data(factor * self.data)
    self.set_z_unit(new_unit)
  
  def change_xy_unit(self, new_unit):
    '''
    Changes the x/y unit to a new one and corrects the ``self.lx``, ``self.ly``, ``self.xc`` and ``self.yc``.
    '''
    factor = self.get_xy_factor() / self.get_xy_factor(new_unit)
    self.set_lx(factor * self.lx)
    self.set_ly(factor * self.ly)
    self.xc = self.xc * factor
    self.yc = self.yc * factor
    self.set_xy_unit(new_unit)
  
  def get_xlim(self):
    '''
    Returns the x limits of the image.
    '''
    return -self.xc, self.lx - self.xc
    
  def get_ylim(self):
    '''
    Returns the y limits of the image.
    '''
    return -self.yc, self.ly - self.yc
  
  def get_xyz(self):
    '''
    Returns useable data from the image as 3 arrays using cartesian coordinates:
    
    * X is the values of x for each pixel.
    * Y is the values of y for each pixel.
    * Z is the values of z for each pixel.
    
    :rtype: 3 ``numpy.array``
    
    .. plot:: example_code/generic/Spm_image-get_xyz.py
      :include-source: 
    '''
    x = np.linspace(0., self.lx, self.x_samples()) - self.xc
    y = np.linspace(0., self.ly, self.y_samples()) - self.yc
    X, Y = np.meshgrid(x,y)
    Z = copy.deepcopy(self.data)
    return X, Y, Z
  
  def get_rtz(self, nr = None, nt = None, rmax = 'max' ):
    '''
    Returns useable data from the image as 3 arrays using cylindrical coordinates:
    
    * R is the values of the radius r for each pixel.
    * T is the values of the angle theta for each pixel.
    * Z is the values of z for each pixel.
    
    :rtype: 3 ``numpy.array``
    
    .. plot:: example_code/generic/Spm_image-get_rtz.py
      :include-source: 
    '''
    nx, ny = self.x_samples(), self.y_samples()
    n = max(nx, ny)
    if nr == None: nr = n
    if nt == None: nt = n  
    X0, Y0, Z0 = self.get_xyz()
    if rmax == 'max': rmax = ((X0**2 + Y0**2).max())**.5
    if rmax == 'min': 
      rmax = min( X0.max(), -X0.min(), Y0.max(), -Y0.min() )
    r = np.linspace(0., rmax, nr)
    t = np.linspace(0., 2 * np.pi, nt)
    R, T = np.meshgrid(r, t)
    X = R * np.cos(T)
    Y = R * np.sin(T)
    X0 = X0.flatten()
    Y0 = Y0.flatten()
    Z0 = Z0.flatten()
    points = np.array([X0,Y0]).transpose()
    Z = interpolate.griddata(points, Z0, (X, Y), method = 'linear')
    return R, T, Z
    
    
  def get_labels(self):
    '''
    Returns labels using LaTeX syntax usable in Matplotlib:
    
    * x label
    * y label
    * z label
    
    :rtype: 3 strings 
    '''
    xlabel = r'${0}$'.format(self.xy_unit.replace('u', '\mu '))
    ylabel = r'${0}$'.format(self.xy_unit.replace('u', '\mu '))
    zlabel = r'${0}$'.format(self.z_unit.replace('u', '\mu '))
    return xlabel, ylabel, zlabel
  
  def get_min_position(self):
    '''
    Returns the position in self.data of the minimum of self.data.
    '''
    Z = self.data
    loc = np.where(Z == Z.min())
    x, y = loc[0][0], loc[1][0]
    return x,y
  
  def get_max_position(self):
    '''
    Returns the position in self.data of the maximum of self.data.
    '''
    Z = self.data
    loc = np.where(Z == Z.max())
    x, y = loc[0][0], loc[1][0]
    return x, y
  
  def plane_fit(self,mask = None):
    '''
    Performs a plane fit on the image and returns the image minus the the plane and the plane itself ``spym.generic.spm_image`` instances.
    
    :param mask: None if no mask or a ``numpy.array`` instance containing np.nan where the data has to be occulted.
    :type mask: None or ``numpy.array``
    :rtype: 2 ``spym.generic.spm_image`` instance
    
    .. plot:: example_code/generic/Spm_image-plane_fit.py
      :include-source: 
    '''
    X,Y,Z =self.get_xyz()
    Zm = copy.copy(Z)
    if mask != None: Zm += (mask-mask)
    coor = np.where(np.isnan(Zm) == False)
    x = X[ coor ] 
    y = Y[ coor ]
    z = Z[ coor ]  
    fitfunc = lambda p, x,y: p[0]*x+p[1]*y+p[2] # Target function
    errfunc = lambda p, x, y,z: fitfunc(p, x,y) - z # Error function
    p0 = [0., 0., 0.]
    p1,success =  optimize.leastsq(errfunc, p0[:], args=(x,y,z))
    P = fitfunc(p1, X, Y)
    Z2 = Z - P
    out = copy.deepcopy(self)
    out.data = Z2
    residual = copy.deepcopy(self)
    residual.data = P
    return out, residual
  
  
  
  def line_fit(self,mask = False, direction = 'x'):
    '''
    Performs a line fit on every line in a given scan direction.
    
    :param mask: array containing ``numpy.nan`` where the data is to be occulted.
    :type mask: ``numpy.array``
    :param direction: 'x' for fits/scans along x axis and "y" for y axis.
    :type direction: string
    :rtype: 2 ``spym.generic.spm_image`` instance
    
    .. plot:: example_code/generic/Spm_image-line_fit.py
      :include-source: 
    '''
    
    if direction not in ['x', 'y']:
      raise Exception('direction must be "x" or "y".')  
    X,Y,Z =self.get_xyz()
    out = copy.deepcopy(self)
    residual = copy.deepcopy(self)
    if direction == 'y':
      X = X.transpose()
      Y = Y.transpose()
      Z = Z.transpose()
      out.data = out.data.transpose()
      residual.data = residual.data.transpose() 
    Zm = Z + (mask-mask)
    for i in xrange(len(Z)):
      x, z, zm = X[i], Z[i], Zm[i]
      zc = copy.copy(zm)
      coor = np.where(np.isnan(zc) == False)
      z_crop = z[coor]
      x_crop = x[coor]
      fitfunc = lambda p, x: p[0]*x+p[1] # Target function
      errfunc = lambda p, x,z: fitfunc(p, x) - z # Error function
      p0 = [0., 0.]
      p1,success =  optimize.leastsq(errfunc, p0[:], args=(x_crop,z_crop))
      l = fitfunc(p1, x)
      z = z - l
      out.data[i] = z
      residual.data[i] = l
    if direction == 'y':
      out.data = out.data.transpose()
      residual.data = residual.data.transpose() 
    return out, residual
  
  def paraboloid_fit(self,mask = None):
    """
    Performs a parabolic fit on the sum of all lines along a given scan direction. The image has to be line_fitted of plane_fitted before applying this tool.
    
    :param mask: array containing ``numpy.nan`` where the data is to be occulted.
    :type mask: ``numpy.array``
    :rtype: 2 ``spym.generic.spm_image`` instance
    
    .. plot:: example_code/generic/Spm_image-line_fit.py
      :include-source: 
    """
    X,Y,Z =self.get_xyz()
    Zm = copy.copy(Z)
    if mask != None: Zm += (mask-mask)
    coor = np.where(np.isnan(Zm) == False)
    x = X[ coor ] 
    y = Y[ coor ]
    z = Z[ coor ]  
    fitfunc = lambda p, x,y: p[0]*x**2 + p[1] *x +p[2]*y**2 + p[3]*y + p[4] # Target function
    errfunc = lambda p, x, y,z: fitfunc(p, x,y) - z # Error function
    p0 = [0., 0., 0., 0., 0.]
    p1,success =  optimize.leastsq(errfunc, p0[:], args=(x,y,z))
    P = fitfunc(p1, X, Y)
    Z2 = Z - P
    out = copy.deepcopy(self)
    out.data = Z2
    residual = copy.deepcopy(self)
    residual.data = P
    return out, residual
  
  def rotate(self, angle, tx = 0., ty = 0., method = 'linear'):
    '''
    Applies a combination of translation and rotation to the image in the xy plane using user selected method. The rotation is centered on the center of the image defined by the ``set_center`` method. This method is based on ``scipy.interpolate.griddata``. The returned image has the same size as the original image. As a consequence, corners can be lost in the process and masked zones can appear as well. Translation is applied before the rotation.
    
    :param angle: angle or rotation in degrees.
    :type angle: float
    :param tx: tranlation along x axis.
    :type tx: float
    :param ty: tranlation along y axis.
    :type ty: float
    :rtype: ``spym.generic.spm_image`` instance
    
    .. plot:: example_code/generic/Spm_image-rotate.py
      :include-source: 
    '''
    angle = np.radians(angle)
    X, Y, Z = self.get_xyz()
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    points = np.array([X+tx,Y+ty]).transpose()
    X2 = np.cos(angle) * X + np.sin(angle) * Y
    Y2 = np.cos(angle) * Y - np.sin(angle) * X
    Z2 = interpolate.griddata(points, Z, (X2, Y2), method = method)
    Z2 = np.reshape(Z2, (self.x_samples(), self.y_samples()))
    out = copy.copy(self)
    out.data = Z2
    return out
  
  def interpolate(self, X, Y, method = 'linear'):
    '''
    Interpolates data on any points arranged in any way (single points, line/1D arrays, zone/2D arrays). Input is just 2 arrays of any identical shape representing x and y coordinates of the points where data has to be interpolated.
    
    :param X: array of x positions
    :type X: ``numpy.array`` 
    :param Y: array of Y positions
    :type Y: ``numpy.array`` 
    :param method: method of interpolation used in ``scipy.interpolate.griddata``
    :type method: string
    
    .. plot:: example_code/generic/Spm_image-interpolate.py
      :include-source: 
    '''
    X0, Y0, Z0 = self.get_xyz()
    X0 = X0.flatten()
    Y0 = Y0.flatten()
    Z0 = self.data.flatten()
    points = np.array([X0,Y0]).transpose()
    Z = interpolate.griddata(points, Z0, (X, Y), method = 'linear')
    mask = Z - Z 
    return X + mask , Y + mask, Z
  
      
  def translate(self, nx, ny):
    '''
    Translates the image by a number of pixels on x and y axis. This methods differs from rotate because it does not involve interpolation and so it is faster but does not allow all translation (just integer number of pixels).
    
    :param nx: number of pixels along x.
    :type nx: signed int
    :param ny: number of pixels along y.
    :type ny: signed int
    :rtype: ``spym.generic.spm_image`` instance
    
    .. plot:: example_code/generic/Spm_image-translate.py
      :include-source: 
    '''
    nx = -nx
    ny = -ny
    Z = self.data
    Z2 = np.zeros_like(Z)
    Z2[:,:] = np.nan
    Nx, Ny = self.x_samples(), self.y_samples() 
    ix0 = max(min(nx, Nx), 0) 
    ix1 = max(min(Nx, Nx + nx), 0) 
    iy0 = max(min(ny, Ny), 0) 
    iy1 = max(min(Ny, Ny + ny), 0)  
    Zs = Z[iy0:iy1, ix0:ix1]     
    Z2[Ny - iy1:Ny -iy0,Nx - ix1:Nx -ix0] = Zs
    out = copy.copy(self)
    out.data = Z2
    return out
 
  def crop(self, cx0 = 0, cx1 = 0, cy0= 0, cy1 = 0):
    '''
    Crops the image by a given pixel number on each side and refreshes image data such as ``self.lx`` and ``self.ly``. Negative values are allowed and produce empty (masked) areas.
    
    :param cx0: number of pixels to remove on lower x side.
    :type cx0: integer
    :param cx1: number of pixels to remove on upper x side.
    :type cx1: integer
    :param cy0: number of pixels to remove on lower y side.
    :type cy0: integer
    :param cy1: number of pixels to remove on upper y side.
    :type cy1: integer
    
    .. plot:: example_code/generic/Spm_image-crop.py
      :include-source: 
    ''' 
    X, Y, Z = self.get_xyz()
    nx0, ny0 =  self.x_samples(), self.y_samples()
    nx = nx0 - cx0 - cx1 # new x size
    ny = ny0 - cy0 - cy1 # new y size
    new_data = np.empty((ny, nx))
    new_data[:] = np.nan
    cropped = Z[ max(cy0, 0) : min(ny0, ny0-cy1 ), max(cx0, 0) : min(nx0, nx0-cx1 ) ]
    new_data[ max(0, -cy0) : max(0, -cy0) + len(cropped),  max(0, -cx0) : max(0, -cx0) + len(cropped[0])] = cropped
    out = copy.deepcopy(self)
    out.data = new_data
    out.lx = self.lx * float(nx) / float(nx0)
    out.ly = self.ly * float(ny) / float(ny0)
    out.xc = self.xc - float(cx0) / float(nx0) * self.lx
    out.yc = self.yc - float(cy0) / float(ny0) * self.ly
    return out

  def center_min(self):
    '''
    Centers the image on the minimum z value.
    '''
    X, Y, Z = self.get_xyz()
    loc = self.get_min_position()
    xc, yc = X[loc], Y[loc] 
    self.set_center(xc = xc, yc = yc)

  def center_max(self):
    '''
    Centers the image on the maximum z value.
    '''
    X, Y, Z = self.get_xyz()
    loc = self.get_max_position()
    xc, yc = X[loc], Y[loc] 
    self.set_center(xc = xc, yc = yc)

  def copy(self):
    '''
    Returns a deepcopy of the image.
    
    :rtype: ``spym.Spm_image`` instance.
    '''
    return copy.deepcopy(self)
  
  def is_like(self, other):
    '''
    Tests if lx, ly, x_samples(), y_samples(), xc, yc are identical and returns the result (True or False). 
    '''
    out = True
    other = other.copy()
    other.change_xy_unit(self.xy_unit)
    other.change_z_unit(self.z_unit) 
    if self.lx != other.lx: out = False
    if self.ly != other.ly: out = False
    if self.x_samples() != other.x_samples(): out = False
    if self.y_samples() != other.y_samples(): out = False 
    if self.xc != other.xc: out = False
    if self.yc != other.yc: out = False
    return out
  
  def pixel_area(self):
    '''
    Returns the pixel area as xy_unit**2.
    '''
    dx = self.lx / (self.x_samples() -1)
    dy = self.ly / (self.y_samples() -1)
    return dx * dy 
  
  def dump2gsf(self, name = None, title = None):
    '''
    Dumps the image to a given file in GSF format. If no file is specified, a string is returned. 
    
    ..note:: Due to Gwyddion's behavior, units are all set to "m".
    
    :param name: name or path to file. If None (default), a string will be returned.
    :type name: str
    :param title: title of the image that will be saved in the GSF file.
    :type title: str
    :rtype: None or str
    
    
    '''
    im = copy.copy(self)
    im.change_xy_unit('m')
    im.change_z_unit('m')
    out = ''
    xy_unit, z_unit = im.xy_unit, im.z_unit
    xy_unit.replace('u', u'\u03bc') # Changing u to mu
    z_unit.replace('u', u'\u03bc')  # Changing u to mu
    if title == None: title = 'Spym Image'
    header = '''Gwyddion Simple Field 1.0
XRes    = {0}
YRes    = {1}
XReal   = {2}
YReal   = {3}
XOffset = {4}
YOffset = {5}
XYUnits = {6}
ZUnits  = {7}
Title   = {8}\n'''.format(
    im.x_samples(), 
    im.y_samples(), 
    im.lx, 
    im.ly, 
    -im.xc, 
    -im.yc, 
    xy_unit, 
    z_unit, 
    title)
    out += header
    # Null padding
    l = len(header)
    k = l//4 * 4 + 4 - l
    if k == 0: k = 4
    out += k * '\0'
    # Binary data
    Z = im.data
    a = Z.flatten().tolist()
    data = ''
    for x in a: data += struct.pack('<f', x) # Conversion to float32 in little endian binary
    out += data
    if name == None:
      return out
    else:
      f = open(name , 'wb')
      f.write(out)
      f.close()
  
 
    
def protractor(ax, r0, r1, nGrad = 180, xc = 0., yc=0., ls = '-', lw = .1, fs = 4., deportText = 0.06):
  '''
  Add a protractor to a subplot in order to measure various angles.
  
  :param ax: subplot where to add the protractor.
  :type ax: ``matplotlib.axes.AxesSubplot`` instance.
  :param r0: min radius of the protractor.
  :type r0: float 
  :param r1: max radius of the protractor.
  :type r1: float
  :param nGrad: number of graduations.
  :type r0: int
  :param xc: x coordinate of the center of the protractor.
  :type xc: float  
  :param yc: y coordinate of the center of the protractor.
  :type yc: float  
  :param ls: line style.
  :type ls: float
  :param lw: line width.
  :type lw: float
  :param fs: font size.
  :type fs: float
  :param deportText: text deportation.
  :type deportText: float
  
  .. plot:: example_code/generic/protractor.py
     :include-source:   
  '''
  for i in xrange(nGrad):
    angle = i * 2. * np.pi / nGrad
    X = [xc + r0*np.cos(angle), xc+ r1*np.cos(angle)]
    Y = [yc + r0*np.sin(angle), yc+ r1*np.sin(angle)]
    ax.plot(X,Y,
      color = 'black', 
      linestyle= ls, 
      linewidth = lw)
    ax.text(
      yc + (X[1]- yc)*(1+deportText), 
      xc + (Y[1]- xc)*(1+deportText), 
      '{0:.1f}'.format(np.degrees(angle)), 
      horizontalalignment='center',  
      verticalalignment='center', 
      size = fs, 
      rotation = np.degrees(angle))  

def load_from_gsf(file_name):
  '''
  Builds a spm_image instance from a Gwyddion Simple File (GSF) file.
  '''
  out = Spm_image()
  f = open(file_name, 'rb')
  lines = f.readlines()
  f.close()
  lines.pop(0) # removing header
  # First we find out the length of the param header:
  i = 0
  while True:
    line = lines[i]
    try:
      line.decode('ascii')
      #if '=' not in line: break
      if line[0] == '\0': break
      i+= 1
    except:
      break
  known_fields = [ 'XRes', 'YRes', 'XReal', 'YReal', 'XOffset', 'YOffset', 'Title', 'XYUnits', 'ZUnits']  
  for j in xrange(i):
    line = lines[j]
    words = line.split('=')
    field = words[0].replace(' ', '')
    value = words[1]
    while value[0]  == ' ': value = value[1:]
    while value[-1] == ' ': value = value[:-1]
    if field in known_fields:
      if field == 'XRes':    x_samples = int(value)
      if field == 'YRes':    y_samples = int(value)
      if field == 'XReal':   out.lx = float(value) 
      if field == 'YReal':   out.ly = float(value)
      if field == 'XOffset': out.xc = - float(value) 
      if field == 'XYUnits': out.xy_units = field.replace(u'\u03bc','u') # Changing mu to u
      if field == 'ZUnits': out.z_units = field.replace(u'\u03bc','u') # Changing mu to u
  data = np.zeros([y_samples * x_samples])
  f = open(file_name, 'rb')
  raw_data = f.read()
  f.close()
  binary_data = raw_data[-y_samples * x_samples * 4:]
  for i in xrange(len(data)):
    data[i] = struct.unpack('<f', binary_data[4*i:4*i+4])[0]
  out.data = np.reshape(data, (y_samples, x_samples))    
  return out

  
