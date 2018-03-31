"""Setup script for object_detection."""

from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ['Pillow>=1.0', 'matplotlib', 'Cython>=0.28.1', 'tensorflow>=1.5']

setup(
    name='object_detection',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
#    dependency_links=['https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI'],
    packages=[p for p in find_packages() if p.startswith('object_detection')],
    description='Tensorflow Object Detection Library',
)
