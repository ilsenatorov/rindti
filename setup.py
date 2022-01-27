import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(
    name="RINDTI",
    version="1.0.0",
    author="Ilya Senatorov",
    author_email="ilya.senatorov@helmholtz-hips.de",
    description="Drug-Target prediction using residue interaction networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ilsenatorov/rindti",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
