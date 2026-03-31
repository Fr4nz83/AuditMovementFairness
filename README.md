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


## Structure of the repository

The repository has the following structure:

- the root folder contains the notebooks that implement our approach and the experimental evaluation
- the ```src``` folder contains the various classes and functions used by the notebooks.
- the ```data_simulator``` folder contains the simulator's original movement data. The folder is also used by various notebooks to store the various information derived from the movement data, i.e., compressed trajectories, stop and move segments, materialized grids, object-to cells mappings, and generated candidates.
- the ```experiments``` folder contain the various groups of datasets used during the experimental evaluation. The folder is also used by the notebooks to store the results obtained during the evaluation.


## How to run our assessment approach

We implemented our assessment approach as a sequence of Jupyter Notebooks that must be executed in order. We detail each of these notebook below, in the order of execution.

**1 - Parse&Compress trajectories of PatternsOfLife Simulator.ipynb**: this notebook parses the Patterns of Life simulator's movement data stored in a set of ```.tsv``` files, and stores them into a single large ```.parquet``` file. It then subsequently compresses the movement data in the parquet file, producing a new noticeably smaller parquet file. Note that the reader can skip the parsing operation, as we already provide the parquet file with the original movement data.

**2 - Trajectory Dataset Stop Detection.ipynb**: this notebook takes as input a dataset of trajectories, and detects their stop segments. Note that the execution of this notebook can be skipped, as we also provide a parquet file containing the stop segments detected from the compressed trajectories. Finally, note that the stay detection algorithm can be configured by customizing the ```min_minutes_stop``` and ```max_radius_stop_meters``` variables. The notebook writes its output in the ```data_simulator``` folder.

**3 - Trajectory Dataset Stop Detection.ipynb**: this notebook detects the move segments from a dataset of trajectories and a dataset of stop segments previously detected from them. Note that the execution of this notebook can be skipped, as we also provide a parquet file containing the move segments. The notebook writes its output in the ```data_simulator``` folder.

**4 - User-to-cells mapping.ipynb**: this notebook implements the step that materializes a set of uniform grids with square cells according to a given set of resolutions and alignments. Note that this step can be customized by modifying the ```set_resolutions``` and ```num_alignments``` variables in the notebook. This notebook also implements the "object-to-cells" step, immediately following the former: for each materialized grid, the notebook proceeds to map every object to the top-k cells that characterize its movement patterns, i.e., the object's "cellset". Note that also this step can be customized by modifying the ```top_k_cells_user``` variable. Finally, note that this step is executed in parallel w.r.t. the grids. The notebook writes its output in the ```data_simulator``` folder.

**5 - Subset of cells to test selection.ipynb**: this notebook implements the candidate generation phase, i.e., it finds out subsets of cells that have at least one associated object. Note that this step is executed in parallel w.r.t. the grids. The notebook writes its output in the ```data_simulator``` folder.

**6 - Filter and flatten candidates.ipynb**: this notebook prepares the candidates generated by the previous notebook for spatial scan statistics processing -- more precisely, it flattens the various data structures that contain the candidates.  The notebook writes its output in the ```data_simulator``` folder.


