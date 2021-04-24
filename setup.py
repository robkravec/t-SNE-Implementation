from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(name='tsne663',
      version='0.3',
      description='Implentation of the t-SNE algorithm for dimension reduction',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/robkravec/tsne663',
      author='Marc Brooks, Rob Kravec, Steven Winter',
      author_email='marc.brooks@duke.edu, robert.kravec@duke.edu, steven.winter@duke.edu',
      license='MIT',
      packages=find_packages(), 
      install_requires = ['matplotlib', 'numpy', 'numba', 'tqdm', 'scikit-learn', 'markdown', 'plotly', 'mpl_toolkits', 'textwrap'])