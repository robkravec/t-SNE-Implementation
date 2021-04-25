from setuptools import setup, find_packages

setup(name='tsne663',
      version='0.8',
      description='Implentation of the t-SNE algorithm for dimension reduction',
      long_description = 'Please see https://github.com/robkravec/STA663_Project for more information on this package',
      url='https://github.com/robkravec/STA663_Project',
      author='Marc Brooks, Rob Kravec, Steven Winter',
      author_email='marc.brooks@duke.edu, robert.kravec@duke.edu, steven.winter@duke.edu',
      license='MIT',
      packages=find_packages(exclude=['tests', 'test.*']),
      install_requires=['matplotlib>=3.1.1', 'numpy>=1.18.1', 'numba>=0.48.0', 'tqdm>=4.42.0', 'scikit-learn>=0.22.1'],
      py_modules=['tsne', 'sim'])