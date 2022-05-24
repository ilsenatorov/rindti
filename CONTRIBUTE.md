# RINDTI contribution guide

The `pyproject.toml` file defines the formatting guidelines for the project.

Please use [pre-commit](https://pre-commit.com/) to run the linter and the tests before committing.

Pre-commit is already included in the installation conda enviornment, so once you have the conda environment activated, you can run `pre-commit install` to run the linter and the tests.

After that, on every commit the code will be formatted and some basic checks will be run.

The commit will be rejected if the documentation coverage is below 80% (this is important for the automatic documentation building).
