# Roadrunner-CellCounter 
A cell counter using computer vision techniques. 

[![Tests](https://github.com/brewinn/Roadrunner-CellCounter/actions/workflows/tests.yml/badge.svg)](https://github.com/brewinn/Roadrunner-CellCounter/actions/workflows/tests.yml)
## Description

A work-in-progress automated cell counter. Ultimately, the program will receive
z-stacked microscope images as input, and return the cell count as output,
possibly along with the set of points designating the cells in the image.

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

A simple convolutional neural network (CNN) model may be run with `python3
cell_counter/cnn_cellcounter.py`, which will generate and train a CNN model on
the synthetic image dataset, then display a graph along with some statistics.

Some tests have been implemented. They may be run with `python3
run_tests.py`.

## Development

Initial development will utilize a two-dimensional synthetic cell dataset
([Synthetic cells](https://bbbc.broadinstitute.org/BBBC005/)) to provide an
abundance of training and testing instances for greater ease of development.
Once either the initial methods have been satisfactorily implemented or the
project deadline passes, the focus will return to more realistic datasets.

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

- Dataset importation: The synthetic-cell dataset, when unzipped to the correct
  place, may be loaded in with a single function call.
- Dataset preprocessing: A method from preprocessing the data has been
  implemented. It reduces the resolution of the images, and normalizes them to
  a [0,1] scale.
- Basic CNN model: A relatively simple model that can be built and run in about
  two minutes.

## To-do

- [X] Dataset reading
- [X] Preprocessing of datasets
    - One preprocessing method has been implemented, and may be modified as
      necessary
- [ ] Implementation of methods
    - [X] Basic CNN
    - [ ] Other models
- [ ] Result collection and analysis

## How to Contribute

The project is not currently open to outside contributors.

This project is conceived as part of the AI class for UTSA. This prohibits
additional contributors during the first few months of the project. Once this
period has passed, the project will be open to more collaborators.

## Project Status

Work-in-progress. Basic features are on the way.
