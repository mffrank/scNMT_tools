from setuptools import setup

setup(name='scnmttools',
      version='0.1',
      description='Convenience functions and data handling stuff for scNMT in python',
      url='https://github.com/mffrank/scNMT_tools',
      author='Max Frank, Rene Snajder',
      license='MIT',
      packages=['scnmttools'],
      install_requires= [
          'numpy',
          'pandas',
          'scipy',
          'anndata'
      ],
      zip_safe=False)
