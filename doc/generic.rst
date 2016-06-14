Generic
========

Generic SPM image processing tools.

Spm_image
~~~~~~~~~~

.. autoclass:: spym.generic.Spm_image
  
Set and load attributes
________________________

.. automethod:: spym.generic.Spm_image.set_name
.. automethod:: spym.generic.Spm_image.set_channel
.. automethod:: spym.generic.Spm_image.set_lx
.. automethod:: spym.generic.Spm_image.set_ly
.. automethod:: spym.generic.Spm_image.set_center
.. automethod:: spym.generic.Spm_image.center_min
.. automethod:: spym.generic.Spm_image.center_max
.. automethod:: spym.generic.Spm_image.set_xy_unit
.. automethod:: spym.generic.Spm_image.set_z_unit
.. automethod:: spym.generic.Spm_image.set_data
.. automethod:: spym.generic.Spm_image.load_from_file

Manipulate image attributes
____________________________

.. automethod:: spym.generic.Spm_image.change_z_unit
.. automethod:: spym.generic.Spm_image.change_xy_unit

Get image data
_______________
.. automethod:: spym.generic.Spm_image.get_labels
.. automethod:: spym.generic.Spm_image.x_samples
.. automethod:: spym.generic.Spm_image.y_samples
.. automethod:: spym.generic.Spm_image.get_xyz
.. automethod:: spym.generic.Spm_image.get_rtz
.. automethod:: spym.generic.Spm_image.get_xlim
.. automethod:: spym.generic.Spm_image.get_ylim
.. automethod:: spym.generic.Spm_image.get_max_position
.. automethod:: spym.generic.Spm_image.get_min_position
.. automethod:: spym.generic.Spm_image.copy
.. automethod:: spym.generic.Spm_image.pixel_area


Artefact and tilt corrections
______________________________

.. automethod:: spym.generic.Spm_image.plane_fit
.. automethod:: spym.generic.Spm_image.line_fit

Image modification
____________________

.. automethod:: spym.generic.Spm_image.rotate
.. automethod:: spym.generic.Spm_image.translate
.. automethod:: spym.generic.Spm_image.interpolate
.. automethod:: spym.generic.Spm_image.section
.. automethod:: spym.generic.Spm_image.crop

Tests
_______
.. automethod:: spym.generic.Spm_image.is_like

Miscellaneous
_____________
.. autofunction:: spym.generic.protractor


