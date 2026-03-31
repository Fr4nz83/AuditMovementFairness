# Assessing Predictive Models for Fairness Based on Movement Patterns

This repository contains the code behind the paper "Assessing Predictive Models for Fairness Based on Movement Patterns".

In this repository you will find:

- Indications on how to generate synthetic movement data with the Patterns of Life simulator;
- The source code used to implement the assessment approach described in the paper;
- The source code used to generate the synthetic unfair auditable datasets used in the experimental evaluation;
- The source code used to conduct the experimental evaluation.


## How to generate the synthetic movement data

We generate a dataset of syntetic trajectories with the [Patterns of Life simulator](https://github.com/onspatial/generate-mobility-dataset). The simulator's configuration has been modified such that it generates the movement data of 100,000 agents moving within the city of Atlanta, Georgia, USA, with a sampling rate of 2 minutes. The seed that has been used for reproducibility purposes has been set to 2.

The simulator stores the generated movement data in very large .tsv files. Those that we use in our experimental evaluation  occupy 225.56 GiB overall. Hence, to facilitate the reader, we made available the movement data used in the experimental evaluation in a parquet file, which can be opened with, e.g., the Python pandas library. Although compressed, the dataframe stored within the parquet file is still very large, hence we recommend to open it in a system with enough RAM.


## How to set up the Python environment to run our assessment approach

Our repository contains a YAML file, ```geo.yaml```, that specifies all the dependencies needed to set up an appropriate conda environment. Assuming that conda is installed, please run ```conda env create -f geo.yaml```, which will create the "geo" environment required to run our approach and recreate our experimental evaluation.


## How to run our assessment approach

We implemented our assessment approach as a sequence of Jupyter Notebooks that must be executed in order. We detail each of these notebook below, in the order of execution.

**1 - Parse&Compress trajectories of PatternsOfLife Simulator.ipynb**: this notebook parses the Patterns of Life simulator's movement data stored in a set of ```.tsv``` files, and stores them into a single large ```.parquet``` file. It then subsequently compresses the movement data in the parquet file, producing a new noticeably smaller parquet file. Note that the reader can skip the parsing operation, as we already provide the parquet file with the original movement data.

**2 - Trajectory Dataset Stop Detection.ipynb**: this notebook takes as input a dataset of trajectories, and detects their stop segments. Note that the execution of this notebook can be skipped, as we also provide a parquet file containing the stop segments detected from the compressed trajectories.

**3 - Trajectory Dataset Stop Detection.ipynb**: this notebook detects the move segments from a dataset of trajectories and a dataset of stop segments previously detected from them. Note that the execution of this notebook can be skipped, as we also provide a parquet file containing the move segments.

**4 - User-to-cells mapping.ipynb**: this notebook implements the step that materialize a set of uniform grids with square cells according to a given set of resolutions and alignments. Note that this step can be customized by modifying the ```set_resolutions``` and ```num_alignments``` variables. This notebook subsequently implements also the "object-to-cells" step: for each materialized grid, the notebook proceeds to map every object to the subset of cells that characterize its movement patterns. Note that also this step can be customized by modifying the ```top_k_cells_user``` variable.  
