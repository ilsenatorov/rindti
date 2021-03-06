import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("rindti/version.py") as infile:
    exec(infile.read())

setuptools.setup(
    name="RINDTI",
    version=version,
    author="Ilya Senatorov",
    author_email="ilya.senatorov@helmholtz-hips.de",
    description="Drug-Target prediction using residue interaction networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ilsenatorov/rindti",
    packages=setuptools.find_packages(),
    keywords=[
        "deep-learning",
        "pytorch",
        "drug-target-interaction",
        "residue-interaction-network",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=">=3.8",
)
