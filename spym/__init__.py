'''
SPyM Documentation
=================================

SPyM is a Python package dedicated to process scanning probe miscroscopy images (SPM). 

.. codeauthor:: Ludovic Charleux <ludovic.charleux@univ-savoie.fr>
    

Installation can be performed in many ways, here a two:
  
* The right way:
  
.. code-block:: bash

   pip install git+https://github.com/lcharleux/spym.git

* If you are contributing to the module, you can just clone the repository:
    
.. code-block:: bash

   git clone https://github.com/lcharleux/spym.git   

And remember to add the abapy/abapy directory to your ``PYTHONPATH``. For example, the following code can be used under Linux (in ``.bashrc`` or ``.profile``):

.. code-block:: bash

  export PYTHONPATH=$PYTHONPATH:yourpath/spym 

SPYM tools rely on the Gwyddion Simple File (GSF) format to read SPM data. Gwyddion can be used to converty nearly any format to GSF. Batch operations can be perfomed by adding the following script to:

.. code-block:: bash

  touch ~/.gwyddion/pygwy/export2gsf.py

.. code-block:: python

  import gwy

  plugin_menu = "/Export all to GSF"
  plugin_type = "PROCESS"
  plugin_desc = "Export all files to GSF format"

  def run():
    cons = gwy.gwy_app_data_browser_get_containers()
    # iterate thru containers and datafields
    for c in cons:
      gwy.gwy_app_data_browser_select_data_field(c, 0)
      filename = c.get_string_by_name("/filename")
      # remove extension from filename
      filebase = filename[0:-4]
      # get directory of datafields where key is key in container
      gwy.gwy_file_save(c, filebase+".gsf", gwy.RUN_IMMEDIATE)
    

.. toctree::
   :maxdepth: 2
'''
