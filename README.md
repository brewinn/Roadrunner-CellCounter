# Roadrunner-CellCounter 
A cell counter using computer vision techniques. 

[![Tests](https://github.com/brewinn/Roadrunner-CellCounter/actions/workflows/tests.yml/badge.svg)](https://github.com/brewinn/Roadrunner-CellCounter/actions/workflows/tests.yml)
## Description

An automated cell counter. This project is for CS-3793/5233 AI
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
- [Future Work](#future-work)
- [Project Status](#project-status)

## Installation

To install, navigate to the project's root directory, and run `pip install
--editable .`. Note that the *requirements.txt* file lists specific versions of
all libraries to use, but different versions of the libraries will likely
function just as well. If running on Windows, several functions will need a
`path` argument, which points to the directory containing the dataset images.

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

The usage of the other models is similar.

Additionally, for use in other scripts, individual models may be imported with 
```
from cell_counter.MODEL import build_MODEL, compile_MODEL, run_MODEL
```
which will bring in the three functions necessary to make and run the models.

Some tests have been implemented. They may be run with `python3
run_tests.py`.

## Development

Development will utilize a two-dimensional synthetic cell dataset
([Synthetic cells](https://bbbc.broadinstitute.org/BBBC005/)) to provide an
abundance of training and testing instances for greater ease of development.

Separate branches and forks have been made to facilitate development, but final
implementations have been brought under this repository.

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
- ResNet model: A deep residual learning based network, based on the work of
  [He \textit{et al.}](https://arxiv.org/pdf/1512.03385.pdf) and code from
  suvooo's
  [Learn-TensorFlow](https://github.com/suvoooo/Learn-TensorFlow/blob/master/resnet/Implement_Resnet_TensorFlow.ipynb).
- U-Net: A fully convolutional network designed for image segmentation, based
  on the work of [Ronneger, Fischer, and
  Brox](https://arxiv.org/pdf/1505.04597.pdf) and code from [Rana *et
  al.*](https://github.com/ashishrana160796/nalu-cell-counting/).
- VGGNet: A network of increasing depth utilizing small convolution filters,
  based on the work of [Simonyan and
  Zissserman](https://arxiv.org/pdf/1409.1556v6.pdf) and code from
  uestcsongtaoli's [vgg\_net](https://github.com/uestcsongtaoli/vgg_net).
- Count-Ception: A network utilizing redundant counting predictions, based on
  the work of [Cohen \textit{et
  al.}](https://github.com/ieee8023/countception).

## To-do

- [X] Dataset reading
- [X] Preprocessing of datasets
- [X] Implementation of methods
- [X] Result collection and analysis

## Future Work

The models have been implemented, but have not been tested over a wide range of
training sizes and conditions due to computational limitations. To achieve a
better idea of each model's strengths and weaknesses, a large variety of
testing and training conditions may be required.


## Project Status

Near completion. The dataset may be imported, filtered, and preprocessed.
Several models have been adapted and implemented. Some final refactoring and
styling are required, but the project is otherwise finished.
