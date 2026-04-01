# Assessing Predictive Models for Fairness Based on Movement Patterns

This repository contains the code behind the paper "Assessing Predictive Models for Fairness Based on Movement Patterns".

In this repository you will find:

- The source code used to implement the assessment approach described in the paper;
- The source code used to generate the synthetic unfair auditable datasets used in the experimental evaluation;
- The source code used to conduct the experimental evaluation;
- A description of how the synthetic movement data was generated with the Patterns of Life simulator;
- How to execute the source code, and how to do it to reproduce the results of the paper's experimental evaluation.


## How to set up the Python environment needed to run our assessment approach

Our repository contains a YAML file, ```geo.yaml```, that specifies all the dependencies needed to set up an appropriate conda environment. Assuming that conda is installed, please run ```conda env create -f geo.yaml```, which will create the "geo" environment required to run our approach and recreate our experimental evaluation.


## Structure of the repository

The repository has the following structure:

- the root folder contains the notebooks that implement our approach and the experimental evaluation
- the ```src``` folder contains the various classes and functions used by the notebooks.
- the ```data_simulator``` folder contains the simulator's original movement data. The folder is also used by various notebooks to store the various information derived from the movement data, i.e., compressed trajectories, stop and move segments, materialized grids, object-to cells mappings, and generated candidates.
- the ```experiments``` folder contain the various groups of datasets used during the experimental evaluation. The folder is also used by the notebooks to store the results obtained during the evaluation.


## How the synthetic movement data was generated

We generate a dataset of syntetic trajectories with the [Patterns of Life simulator](https://github.com/onspatial/generate-mobility-dataset). The simulator's configuration has been modified such that it generates the movement data of 100,000 agents moving within the city of Atlanta, Georgia, USA, with a sampling rate of 2 minutes. The seed that has been used for reproducibility purposes has been set to 2.

The simulator stores the generated movement data in very large .tsv files. Those that we use in our experimental evaluation  occupy 225.56 GiB overall. Hence, to facilitate the reader, we made available the movement data used in the experimental evaluation in a parquet file, which can be opened with, e.g., the Python pandas library. The dataframe stored within the parquet file however takes a lot of RAM once loaded, so please refer to the description below of the notebook numbered **1** about how to use this dataset.


## Where to download the movement data and the intermediate outputs of the approach

To facilitate the reader and speed up the evaluation of the article's results, we have uploaded the synthetic movement data as well as various intermediate outputs produced by our approach during its various steps in an [anonymized Figshare repository](https://figshare.com/s/93b04d0a6128d3e7ca32). Please download the .zip archives from that repository if you want to skip executing any of the steps of our approach, or you want to compare the output produced locally with that used for the results shown in the article.


## How to execute our assessment approach and reproduce the article's experimental evaluation

We implemented our assessment approach as a sequence of Jupyter Notebooks that must be executed in a certain order. We detail what each of these notebooks do and what they output below; we also present them in the expected order of execution. Please note that all the notebooks have been documented so that a reader can understand the operations conducted within them.

***

### Preparing the synthetic movement data

The first set of notebooks deals with **preparing the simulator's movement data for subsequent operations**. 

- **1 - Parse&Compress trajectories of PatternsOfLife Simulator.ipynb**: this notebook parses the Patterns of Life simulator's movement data stored in a set of large ```.tsv``` files, and stores them into a single ```.parquet``` file. It then subsequently compresses the movement data in the parquet file, producing a new noticeably smaller parquet file. 
We report that these two operations require to have a machine with a very large amount of RAM, especially the compression operation which requires around 140 GiB.
Accordingly, the reader can skip both operations, as we already provide the parquet files of both the original and compressed movement data.

- **2 - Trajectory Dataset Stop Detection.ipynb**: this notebook takes as input a dataset of trajectories, and detects their stop segments. Note that the execution of this notebook can be skipped, as we also provide a parquet file containing the stop segments detected from the compressed trajectories. Finally, note that the stay detection algorithm can be configured by customizing the ```min_minutes_stop``` and ```max_radius_stop_meters``` variables. The notebook writes its output in the ```data_simulator``` folder. Note that the reader can skip executing this notebook, as we already provide the parquet file with the stop segments.

- **3 - Trajectory Dataset Move Detection.ipynb**: this notebook detects the move segments from a dataset of trajectories and a dataset of stop segments previously detected from them. Note that the execution of this notebook can be skipped, as we also provide a parquet file containing the move segments. The notebook writes its output in the ```data_simulator``` folder. Note that the reader can ignore this notebook, as the move segments are not used by the current version of our assessment approach.

- **4 - User-to-cells mapping.ipynb**: this notebook implements the step that materializes a set of uniform grids with square cells according to a given set of resolutions and alignments. Note that this step can be customized by modifying the ```set_resolutions``` and ```num_alignments``` variables in the notebook. This notebook also implements the "object-to-cells" step, immediately following the former: for each materialized grid, the notebook proceeds to map every object to the top-k cells that characterize its movement patterns, i.e., the object's "cellset". Note that also this step can be customized by modifying the ```top_k_cells_user``` variable. Note that this step is executed in parallel w.r.t. the grids. The notebook writes its output in the ```data_simulator``` folder. Finally, note that the reader can skip executing this notebook, as we already provide the materialized grids and the mappings in a set of pickle files.

- **5 - Subset of cells to test selection.ipynb**: this notebook implements the candidate generation phase, i.e., it finds out subsets of cells that have at least one associated object. Note that this step is executed in parallel w.r.t. the grids. The notebook writes its output in the ```data_simulator``` folder. Finally, note that the reader can skip executing this notebook, as we already provide the candidates in a set of pickle files.

- **6 - Filter and flatten candidates.ipynb**: this notebook prepares the candidates generated by the previous notebook for spatial scan statistics processing -- more precisely, it flattens the various data structures that contain the candidates.  The notebook writes its output in the ```data_simulator``` folder. Finally, note that the reader can skip executing this notebook, as we already provide the flattened candidates in a pickle file.

This terminates the notebooks that implement the first four steps of our approach, i.e., the steps that focus on the preparation of the simulator's movement data for subsequent steps.

***

### Run the experimental evaluation

Next, we need to perform with the **experimental evaluation**. We execute the following set of notebooks.

- **Exp - Generator of unfair auditable datasets.ipynb**: this notebook generates the various groups of unfair auditable datasets required to run the experiments. To configure the generation process, the user must customize the ```params_datasets``` variable in the notebook. Even in the generation process is massively parallelized w.r.t. the datasets in a group, we report that certain groups of datasets requires a very long execution time to be generated. Hence, in the case a user wishes to execute this notebook to generate their own unfair auditable datasets, we recommend to use a machine with a large number of CPU cores. The notebook writes its output in the ```experiments``` folder. In any case, the reader can skip executing this notebook, as we already provide the unfair datasets used during the experimental evaluation.

- **7 - Assess Datasets for Fairness Based on Movement Patterns.ipynb**: this notebook performs the Bernoulli-based spatial scan statistic considering all the candidate subsets of cells from the various materialized grids, over groups of unfair auditable datasets. The execution can be customized by modifying the following variables: ```num_simulations```, which represents the number of Monte Carlo simulations used to approximate the distribution of the maximum log-likelihood ratio under the null hypothesis; ```alpha```, which represents the required statistical significance level; and ```list_set_groups```, which specifies with set of groups of datasets to consider (e.g., the groups in which we vary the number of objects associated with an hotspot). The notebook outputs the results in the ```experiments``` folder, more precisely in the subfolders contaning the sets of groups of datasets being considered. To facilitate the reader, we have already included the files containing the results of this step, and which have subsequently been used in the article.

***

At this point, we have the results of the Bernoulli-based spatial scan statistic applied over several groups of unfair auditable datasets. Next, we execute the set of notebooks that transform these results into the ones actually shown in the article.


- **8 - Compute stats results group datasets.ipynb**: this notebook transforms the results of the Bernoulli-based spatial scan statistic in formats (i.e., latex tables and CSVs) suitable for our article. Note that the user must customize the variable ```name_set_groups_datasets```, which specifies the particular set of groups of datasets considered by the notebook (e.g., the set in which we vary the number of objects per hotspot). To facilitate the reader, we have included the CSVs, which have subsequently been used to generate the article's plots.

- **9 - Plot results.ipynb**: this notebook generates the plots from the CSVs generated by the previous notebook.

### Exploratory strategy example

Finally, the notebook **10 - Advanced inspection hypothesis test results.ipynb** permits to perform the exploratory strategy described in the article's experimental evaluation on any of the unfair auditable datasets used during the experimental evaluation. Assuming that the various groups of datasets are located in the appropriate paths, the user can customize the notebook by changing the following variables: ```name_set_groups_datasets``` specifies the set of groups of datasets considered. ```idx_selected_group``` specifies the dataset group considered from the set; ```idx_dataset``` specifies the index of the dataset considered from the group.
