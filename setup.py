import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyTorch-Factor_Analysis",
    version="0.0.1",
    author="Aditya Ahuja",
    description="",
    url="~",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)