from setuptools import setup, find_packages

REQUIREMENTS = [
    'torch>=2.2.1',
    'torchvision>=0.17.1',
    'numpy>=1.26.3',
    'pydantic>=2.6.3',
    'pyyaml>=6.0.1',
    'torchmetrics>=1.3.1',
    'pycocotools>=2.0.7',
    'tqdm>=4.66.2',
    'nltk>=3.8.1',
    'pycocoevalcap>=1.2.0',
]

setup(
    name='image_captioning',
    packages=find_packages("src"),
    package_dir={"": "src"},
    version='1.0.0',
    description='Image Captioning using PyTorch',
    author='Surya Mahadi',
    install_requires=REQUIREMENTS,
    python_requires='>=3.10'
)