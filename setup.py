from setuptools import setup, find_packages


MIN_REQUIRED_TORCH_VERSION = "2.0.0"
MIN_REQUIRED_TORCHVISION_VERSION = "0.15.0"
REQUIREMENTS = [
    'numpy>=1.26.3',
    'pydantic>=2.6.3',
    'pyyaml>=6.0.1',
    'torchmetrics>=1.3.1',
    'pycocotools>=2.0.7',
    'tqdm>=4.66.2',
    'nltk>=3.8.1',
    'pycocoevalcap>=1.2.0',
]

def check_if_torch_installed():
    try:
        import torch

        current_version = torch.__version__

        # remove any suffixes (e.g. +cu102 in the case of CUDA 10.2)
        current_version = current_version.split("+")[0]

        if torch.__version__ < MIN_REQUIRED_TORCH_VERSION:
            REQUIREMENTS.append(f"torch>={MIN_REQUIRED_TORCH_VERSION}")
    except ImportError:
        REQUIREMENTS.append(f"torch>={MIN_REQUIRED_TORCH_VERSION}")

def check_if_torchvision_installed():
    try:
        import torchvision

        current_version = torchvision.__version__
        
        # remove any suffixes (e.g. +cu102 in the case of CUDA 10.2)
        current_version = current_version.split("+")[0]

        if torchvision.__version__ < MIN_REQUIRED_TORCHVISION_VERSION:
            REQUIREMENTS.append(f"torchvision>={MIN_REQUIRED_TORCHVISION_VERSION}")
    except ImportError:
        REQUIREMENTS.append(f"torchvision>={MIN_REQUIRED_TORCHVISION_VERSION}")


check_if_torch_installed()
check_if_torchvision_installed()


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