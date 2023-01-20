# Welcome to the Artificial Metabolic Networks repository

This repository is entirely written in **python**. We make use of **jupyter** notebooks,
calling custom functions libraries storing the main objects and functions used in the project. We detail here two ways of using the repo, either on **Colab** or **locally**.

One can clone the git directly in a Google Drive and open the notebooks in Google Colab. This is a good way to make first testings and have a glimpse of the project.

Also, one can clone the git locally and install a **conda** environment we provide, to be used for the project once it's linked to your jupyter environment. This will provide better reproducibility than the colab install. We recommend this option for computationally costly usage of the repository.

A **tutorial** is available as the notebook `Tutorial.ipynb`. This is a good place to start, going through all the detailed steps for building and training an AMN model. This step-by-step exploration of the project will take about 20 minutes to be runned.

Note: For local installs, only Linux (Ubuntu 22.04) and MacOS (Monterey) have been tested, but Windows should work.

## Installation instructions:

### 1) Google Drive/Colab install

This install takes about 3 minutes, then each notebook needs 3 additional minutes to be runned.

- **Clone the repository**

Open this notebook, make a copy in your own Google Drive if you want to make modifications, e.g. the path to which the repo is cloned on the drive (File > Save a copy in Drive) and follow the instructions:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AhGt8LH6MFTNToMD-VgSy25s8AgE5xDg?usp=sharing)


- **Navigate to the root of `amn_release` in your drive**

 And that's it! You will have access to all notebooks. Simply double-click any notebook to open it in colab, and follow the instructions in each of them.

NB: Avoid costly operations in Colab. Also, a fresh environment is created for each notebook opened, expect around 3 minutes of installation each time you open a new notebook. And don't panic if you see the Colab kernel restarting automatically, it's necessary for conda to work in Colab.


### 2) Local install

This install takes between 5 and 15 minutes (if you already connected jupyter and conda together, it will be shorter).

- **Clone** the git ([how to clone a git repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository))

- Install a distribution of **conda** if not already installed ([how to install conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation))

- Import the **environment** `environment_amn.yaml` (stored at the root of the repository) with the following command:

`conda env create -n AMN --file environment_amn.yml`

NB: One can change the name 'AMN' to anything, this will be the name of your created environment.

- Make your conda environment **accessible to jupyter**, if not already the case ([how to get conda environments in jupyter](https://towardsdatascience.com/get-your-conda-environment-to-show-in-jupyter-notebooks-the-easy-way-17010b76e874))

- When opening the project's notebooks, make sure to use the right `kernel` with 'Kernel > Change kernel' in the toolbar.

## Content description:

In this repository you will find different **notebooks** that have different purposes. They are all linked to a python **function-storing file**, except for the `Figures.ipynb` notebook which runs alone. Their purpose is explained hereafter.

Some **folders** store different kinds of datasets, which will be described here.

Finally, independent **files** are in this repository for specific reasons, detailed below. 

### 1) Notebooks and corresponding function files

- Duplicate the two-sided and exchange reactions in a SBML model, with `Duplicate_Model.ipynb` (linked to the functions-storing python file `Duplicate_Model.py`). This notebook shows the step-by-step workflow This is mandatory before performing any neural computations with metabolic networks, so that all fluxes are positive. All steps are shown in the notebook, with details on each step of the process.

- Build a suitable experimental dataset, with `Build_Experimental.ipynb` (linked to the functions-storing python file `Build_Experimental.py`). This notebook shows the step-by-step workflow for generating combinations of variables (in a Design of Experiments fashion) to be tested experimentally, then processing the raw data from plate reader runs, and finally building an appropriate growth rate training set.

- Build *in silico* or *in vivo* (*i.e.* with experimental measures) training sets for AMNs, with `Build_Dataset.ipynb` (linked to the functions-storing python file `Build_Dataset.py`). This notebook shows many examples of training set generations, with *in silico* simulations or *in vivo* datasets. For more detailed instructions and explanations on parameters and methods, refer to the functions-storing file and the `Tutorial.ipynb` notebook.

- Build AMN models, train them and record their performance, with all notebooks starting with `Build_Model_` (linked to the functions-storing python file `Build_Model.py`). These notebooks shows many examples of models generation and training, with *in silico* or *in vivo* training sets. For more detailed instructions and explanations on parameters and methods, refer to the functions-storing file and the `Tutorial.ipynb` notebook. A variety of notebooks are available, each designed for a specific model type. The suffixes correspond to: `MM` for mechanistic models (no learning), `ANN_Dense` for classical dense neural networks, `AMN` for the hybrid models we developed in this project, and `RC` for the reservoir computing framework to use on top of a trained AMN.

- Making figures, with `Figures.ipynb` (standalone jupyter notebook). This notebook simply generates the figures shown in the research paper of the AMN project. It is a standalone notebook that isn't linked to any function-storing file.

NB: All function-storing python files are under the folder `/Library`.

### 2) Data storing folders

- `Dataset_experimental` containing all **experimental** data used for the AMN research paper. It contains raw data (`_data.csv` suffix), companion files for processing the raw data (`_start_stop.csv` and `_compos.csv` suffixes) and processed data (`_results.csv` suffix). It also contains raw compositions generated in a Design of Experiments fashion (`compositions_` prefix). Finally, here is stored the final dataset used in the AMN research paper, called `EXP110.csv`.

- `Dataset_input` containing files for **guiding** the generation of training sets. It contains the models (`.xml` extension) and associated files for guiding the generation of training sets with corresponding models (`.csv` extension). It also contains solutions to be used with cobrapy (when performing reservoir computing, extracting the exchange reactions predicted bounds), for practical reasons. Note that models must be saved since a reduction of the model can be performed in `Build_Dataset.ipynb`.

- `Dataset_model` containing **training sets** (`.npz` extension) and associated model files (`.xml` extension). The filenames are built as follows: name of the metabolic model used to generate the training set + type of bound + number of elements in the training set.

- `Reservoir` contains **trained models** (`.h5` extension) and corresponding model **hyper-parameters** files (`.csv` extension). The filenames are built as follows: name of the metabolic model used to generate the training set + type of bound + number of elements in the training set + model type for learning.

- `Result` contains various **raw data** files used to generate figures in the `Figures.ipynb` notebook. One can refer directly to this notebook to know how each data file is used.

NB: `/Library` is only storing function-storing python files, `/Figures` is only storing figures.

### 3) Independent files

- `README.md` is the file you are reading.

- `LICENSE` gives an MIT licensing to the project.

- `environment_amn.yml` is the file to create an appropriate conda environment for a **local** install. It has exactly the same packages and versions than the environment used to develop this project.

- `environment_amn_light.yml` is the file to create an appropriate conda environment for a **colab** install. It has just a few packages that are not present by default in colab, and needed for this project.
