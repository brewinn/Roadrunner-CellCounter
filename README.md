# Roadrunner-CellCounter 
A cell counter using computer vision techniques. 

[![Tests](https://github.com/brewinn/Roadrunner-CellCounter/actions/workflows/tests.yml/badge.svg)](https://github.com/brewinn/Roadrunner-CellCounter/actions/workflows/tests.yml)
## Description

A work-in-progress automated cell counter. This project is for CS-3793/5233 AI
course of UTSA, taught by Dr. Kevin Desai. It was originally intended to work
on 3D images in collaboration with Dr. Hye Young Lee and team for easier cell
counting, but this project has been delayed, and will continued in a separate
repository.

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
--editable .`. Note that the `requirements.txt` file lists specific versions of
all libraries to use, but different versions of the libraries will likely
function just as well.

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

The usage of `fcrn_cellcounter.py` and `nalu_fcrn_cellcounter.py` is similar.

Some tests have been implemented. They may be run with `python3
run_tests.py`.

## Development

Development will utilize a two-dimensional synthetic cell dataset
([Synthetic cells](https://bbbc.broadinstitute.org/BBBC005/)) to provide an
abundance of training and testing instances for greater ease of development.

Separate branches and forks have been made to facilitate development, but final
implementations will be brought under this repository.

## Credits

The original project developers include Brendan Winn, Amber Beserra, Michael
Ginsberg, Alan Cabrera, and William Wells. This project was made as a part of
CS-3793/5233 AI course of UTSA, taught by Dr. Kevin Desai. 

We would also like to thank Dr. Hye Young Lee and team of UT Health for
providing the inspiration. Development of the 3D cell counter, while delayed,
will continue nonetheless.

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
  a [0,1] scale. Additionally, images with specific qualities (e.g. a low
  amount of blur) can be selected for specifically via a Pandas DataFrame.
- Basic CNN model: A relatively simple model that can be built and run in about
  two minutes.
- FCRN model: A fully convolutional regression network, based on the work of
  [Rana *et al.*](https://github.com/ashishrana160796/nalu-cell-counting/blob/master/research-paper-tex/dual-page-latex-work/dual-page-latex-paper.pdf)
- NALU-FCRN model: A FCRN network augmented with neural arithmetic logic units, based on the work of
  [Rana *et al.*](https://github.com/ashishrana160796/nalu-cell-counting/blob/master/research-paper-tex/dual-page-latex-work/dual-page-latex-paper.pdf)

## To-do

- [X] Dataset reading
- [X] Preprocessing of datasets
- [ ] Implementation of methods
    - [X] Basic CNN
    - [X] FCRN
    - [X] NALU-FCRN
    - [ ] Other models
- [ ] Result collection and analysis

## How to Contribute

This project is conceived as part of the AI class for UTSA. This prohibits
additional contributors during the first few months of the project. Once this
period has passed, the project will be open amendments.

## Project Status

Work-in-progress. Several models have been implemented, and the dataset may be
preprocessed and filtered.
