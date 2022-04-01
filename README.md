# Roadrunner-CellCounter A cell counter using computer vision techniques. 

## Description

A work-in-progress automated cell counter. The program will receive z-stacked
microscope images as input, and return the cell count as output, possibly along
with the set of points designating the cells in the image.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Credits](#credits)
- [License](#license)
- [Features](#features)
- [To-do](#to-do)
- [How to Contribute](#how-to-contribute)
- [Project Status](#project-status)

## Installation

To install, navigate to the project's root directory, and run `pip install
--editable .`.

### Removal

To uninstall, run `pip uninstall cell_counter`.

## Usage

First, download the synthetic-cell dataset from [Synthetic
cells](https://bbbc.broadinstitute.org/BBBC005/). The dataset should then be
unzipped into the *resources* directory. The structure should look like
*resources/BBBC005_v1_images/IMAGES*.

Only a few tests have been implemented so far. They may be run with `python3
tests/test_import_tiff.py`.

## Development

Initial development will utilize a two-dimensional synthetic cell dataset
([Synthetic cells](https://bbbc.broadinstitute.org/BBBC005/)) to provide an
abundance of training and testing instances for greater ease of development.
Once either the initial methods have been satisfactorily implemented or the
project deadline passes, the focus will return to more realistic datasets.
The rest of this section will give a reference to some development guidelines for the
project.

### Git

Git should be used to track changes in code. While not every single change will
need its own commit, major changes need to be committed separately. Be sure to
include an 
[informative commit message](https://www.freecodecamp.org/news/writing-good-commit-messages-a-practical-guide/)!

### Project Structure

Python has its own way of including code from other files. To make it as easy
as possible to install and use, we'll make our project code into a package. A
reference for the structure can be found in this article: 
[The optimal python project structure](https://awaywithideas.com/the-optimal-python-project-structure/)

### Code Formatting

To give a consistent formatting style to the code, we'll make use of
[Black](https://github.com/psf/black). One way to install *Black* is via `pip
install black`. Once installed, you can run *Black* on a single file `black
FILE` or on an entire directory (including subdirectories) via `black
DIRECTORY-TO-FORMAT`.

### Testing

Project code should be made via Test-Driven Development (TDD). TDD is a broad
topic, and you should look into it if you have the time. For our purposes, this
means that tests should be written *before* the relevant code is added, and
then the code modified to make the test pass. Send a message on discord if you
need an introduction.

## Credits

The original project developers include Brendan Winn, Amber Beserra, Michael
Ginsberg, Alan Cabrera, and William Wells. This project was made as a part of
CS-3793/5233 AI course of UTSA, taught by Dr. Kevin Desai. 

We would also like to thank Dr. Hye Young Lee and team of UT Health for
providing the inspiration and data for the project.

The dataset used in the initial development is public domain, and available for download from the
Broad Bioimage Benchmark Collection here: [Synthetic
Cells](https://bbbc.broadinstitute.org/BBBC005/) 

## License

MIT: <https://mit-license.org>

## Features

Below is a list of currently implemented features:

- Nothing just yet!

## To-do

- [ ] Dataset Reading
- [ ] Preprocessing of datasets
- [ ] ...and much more

## How to Contribute

The project is not currently open to outside contributors.

This project is conceived as part of the AI class for UTSA. This prohibits
additional contributors during the first few months of the project. Once this
period has passed, the project will be open to more collaborators.

## Project Status

Work-in-progress. Basic features are on the way.
