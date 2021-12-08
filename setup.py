from setuptools import setup
import os
import sys

_here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(_here, "README.md"), "r", encoding="utf-8") as f:
    long_description = f.read()

with open(os.path.join(_here, "requirements.txt"), "r") as f:
    install_requires = f.read().splitlines()

version = {}
with open(os.path.join(_here, "kchannel", "version.py"), "r") as f:
    exec(f.read(), version)


setup(
    name="kchannel",
    version=version['__version__'],
    author="Chun Kei Lam",
    author_email="chun-kei.lam@mpibpc.mpg.de",
    description="Analysis tool for potassium channels in MD simulations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=['kchannel'],
    python_requires=">=3.6",
    install_requires=install_requires
)
