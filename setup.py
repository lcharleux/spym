from setuptools import setup

setup(name='spym',
      version='0.1',
      description="SPM image processing",
      long_description="",
      author='Ludovic Charleux',
      author_email='ludovic.charleux@univ-savoie.fr',
      license='GPL v2',
      packages=['spym'],
      zip_safe=False,
      install_requires=[
          "numpy",
          "scipy",
          "matplotlib"
          ],
      )
