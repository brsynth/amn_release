# Welcome to the Artificial Metabolic Networks repository

This repository is entirely written in python. We make use of jupyter notebooks,
calling custom functions libraries storing the main objects and functions used in the notebooks. We detail here two ways of using the repo, either on Colab or locally.

One can clone the git directly in a Google Drive environment and open the notebooks in Google Colab. This is a good way to make first testings and examine the project.

Also, one can clone the git locally and install a conda environment we provide, to be used for the project once it's linked to your jupyter environment. This will provide better reproducibility than the colab install. We recommend this option for computationally costly usage of the repository.

A tutorial is available as a notebook: `Tutorial.ipynb`. This is a good place to start, going through all the detailed steps for building an AMN. 

## Installation instructions:

### 1) Google Drive/Colab install

- Clone the repository:

Open this notebook, make a copy in your own Google Drive (File > Save a copy in Drive) and follow the instructions: https://colab.research.google.com/drive/1AhGt8LH6MFTNToMD-VgSy25s8AgE5xDg?usp=sharing 

- Install conda and import the AMN environment 

Open this notebook, make a copy in your own Google Drive (File > Save a copy in Drive) and follow the instructions: 
https://colab.research.google.com/drive/1PxSfXA8NaFz3LbQ1OzOPLCwvk2EvPRrY?usp=sharing

- Navigate to the root of `amn_release` in your drive and that's it! You will have access to all notebooks with the right conda environment.

- Optionally use this notebook to make updates on the git repository:
https://colab.research.google.com/drive/1GyMEHPubIQzaZmUOXLpsjl7R6Hi893Ic?usp=sharing



### 2) Local install

- Clone the git (how to git clone)

- Install conda if not already installed (how to install conda)

- Import the environment `env.yaml` (stored at the root of the repository) with the following command:

`this is the command to import env`

- Add the conda environment to your jupyter environments, if not already the case (following this tutorial)

- When opening the project's notebooks, make sure to activate the right `env` with 'Kernel > Change kernel'

## Content description:

In this repository you will find different **notebooks** that have different purposes. They are all linked to a python **function-storing file**, except for the `Figure.ipynb` notebook which runs alone. Their purpose is explained hereafter.

Some **folders** store different kinds of datasets, they will be described here.

Finally, independent **files** are in this repository for specific reasons, detailed below. 

### 1) Notebooks and corresponding function files

- Duplicate the two-sided and exchange reactions in a SBML model, with `Duplicate_Model.ipynb` (linked to the functions-storing python file `Duplicate_Model.py`). This notebook shows the step-by-step workflow This is mandatory before performing any neural computations with metabolic networks, so that all fluxes are positive. All steps are shown in the notebook, with details on each step of the process.

- Build a suitable experimental dataset, with `Build_Experimental.ipynb` (linked to the functions-storing python file `Build_Experimental.py`). This notebook shows the step-by-step workflow for generating combinations of variables (in a Design of Experiments fashion) to be tested experimentally, then processing the raw data from plate reader runs, and finally building an appropriate growth rate training set.

- Build *in silico* or *in vivo* (*i.e.* with experimental measures) training sets for AMNs, with `Build_Dataset.ipynb` (linked to the functions-storing python file `Build_Dataset.py`). This notebook shows many examples of training set generations, with *in silico* simulations or *in vivo* datasets. For more detailed instructions and explanations on parameters and methods, refer to the functions-storing file and the `Tutorial.ipynb` notebook.

- Build AMN models, train them and record their performance, with `Build_Model.ipynb` (linked to the functions-storing python file `Build_Model.py`). This notebook shows many examples of AMN generation and training, with *in silico* or *in vivo* training sets. For more detailed instructions and explanations on parameters and methods, refer to the functions-storing file and the `Tutorial.ipynb` notebook.

- Making figures, with `Figures.ipynb` (standalone jupyter notebook). This notebook simply generates the figures shown in the research paper of the AMN project.

### 2) Data storing folders

- 

