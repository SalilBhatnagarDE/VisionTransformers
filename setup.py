# from distutils.core import setup
import setuptools
# from setuptools import find_packages

setuptools.setup(
    name="Vision_Transformer",
    include_package_data=True,
    version='0.1',
    author="Salil Bhatnagar",
    author_email="salil.bhatnagar@fau.de",
    packages=setuptools.find_packages(),
    install_requires=["torch", "torchvision", "matplotlib", "matplotlib.pyplot", "numpy", "torch.utils.data", "torch.nn", "torch.nn.functional",
                      "PIL", "torchvision.transforms", "einops", "einops.layers.torch", "cv2", "math"]
)
