# Assessing Predictive Models for Fairness Based on Movement Patterns

This repository contains the code accompanying the paper **"Assessing Predictive Models for Fairness Based on Movement Patterns"**.

In this repository, you will find:

- The source code used to implement the assessment approach described in the paper;
- The source code used to generate the synthetic unfair auditable datasets used in the experimental evaluation;
- The source code used to conduct the experimental evaluation;
- A description of how the synthetic movement data was generated with the Patterns of Life simulator;
- Instructions on how to execute the code and reproduce the results of the paper’s experimental evaluation.

To facilitate reuse and speed up the evaluation of the article's results, we have uploaded the synthetic movement data, as well as various intermediate outputs produced by our approach during its different steps, to an [anonymized Figshare repository](https://figshare.com/s/93b04d0a6128d3e7ca32). Please download the `.zip` archives from that repository if you want to skip executing any of the steps of our approach, or if you want to compare the output produced locally with that used for the results shown in the article. Please note that the `.zip` archives should be decompressed in the **root folder** of this repository. 


## How to set up the Python environment needed to run the assessment approach

This repository contains a YAML file, `geo.yaml`, that specifies all the dependencies needed to create a suitable conda environment. Assuming that conda is installed, please run:

```bash
conda env create -f geo.yaml
```

This will create the `geo_test` environment required to run the approach and reproduce the experimental evaluation.


## Structure of the repository

The repository has the following structure:

- the root folder contains the notebooks that implement our approach and the experimental evaluation;
- the `src` folder contains the various classes and functions used by the notebooks;
- the `data_simulator` folder contains the simulator's original movement data. The folder is also used by various notebooks to store the information derived from the movement data, such as compressed trajectories, stop and move segments, materialized grids, object-to-cells mappings, and generated candidates;
- the `experiments` folder contains the various groups of datasets used during the experimental evaluation. The folder is also used by the notebooks to store the results obtained during the evaluation.


## How the synthetic movement data was generated

We generate a dataset of synthetic trajectories with the [Patterns of Life simulator](https://github.com/onspatial/generate-mobility-dataset). The simulator's configuration was modified so that it generates the movement data of 100,000 agents moving within the city of Atlanta, Georgia, USA, with a sampling rate of 2 minutes. The seed used for reproducibility was set to 2.

The simulator stores the generated movement data in very large `.tsv` files that occupy 225.56 GiB overall. We therefore made the movement data used in our work available as a parquet file, which can be opened with, for example, the Python pandas library. The dataframe stored in the parquet file, however, still requires a large amount of RAM once loaded, so please refer to the description below of notebook **1** for guidance on how to use this dataset.


## How to execute our assessment approach and reproduce the article's experimental evaluation

We implemented our assessment approach as a sequence of Jupyter notebooks that must be executed in a specific order. Below we describe what each notebook does, what it outputs, and the expected order of execution. Please note that all the notebooks have been documented so that readers can understand the operations performed within them.

***

### Preparing the synthetic movement data

The first set of notebooks deals with **preparing the simulator's movement data for subsequent operations**.

- **1 - Parse&Compress trajectories of PatternsOfLife Simulator.ipynb**: this notebook parses the Patterns of Life simulator's movement data stored in a set of large `.tsv` files and stores it in a single `.parquet` file. It then compresses the movement data in that parquet file, producing a noticeably smaller parquet file.
We note that these two operations require a machine with a very large amount of RAM, especially the compression step, which requires around 140 GiB.
Accordingly, readers can skip both operations, as we already provide the parquet files for both the original and compressed movement data (to this end, see the dataset repository).

- **2 - Trajectory Dataset Stop Detection.ipynb**: this notebook takes as input a dataset of trajectories and detects their stop segments. Note that the reader can skip the execution of this notebook, as we also provide a parquet file containing the stop segments detected from the compressed trajectories (to this end, see the dataset repository). Finally, note that the stay-detection algorithm can be configured by customizing the `min_minutes_stop` and `max_radius_stop_meters` variables. The notebook writes its output to the `data_simulator` folder.

- **3 - Trajectory Dataset Move Detection.ipynb**: this notebook detects the move segments from a dataset of trajectories and a dataset of stop segments previously detected from them. Readers can ignore this notebook, as the move segments are not used by the current version of our assessment approach. Note that the execution of this notebook can be skipped, as we also provide a parquet file containing the move segments (to this end, see the dataset repository). The notebook writes its output to the `data_simulator` folder.

- **4 - User-to-cells mapping.ipynb**: this notebook implements the step that materializes a set of uniform grids with square cells according to a given set of resolutions and alignments. This step can be customized by modifying the `set_resolutions` and `num_alignments` variables in the notebook. The notebook also implements the immediately following "object-to-cells" step: for each materialized grid, it maps every object to the top-k cells that characterize its movement patterns, i.e., the object's "cellset". This step can also be customized by modifying the `top_k_cells_user` variable. The step is executed in parallel with respect to the grids. The notebook writes its output to the `data_simulator` folder. Finally, readers can skip executing this notebook, as we already provide the set of materialized grids and mappings in pickle files (to this end, see the dataset repository).

- **5 - Subset of cells to test selection.ipynb**: this notebook implements the candidate-generation phase, i.e., it finds subsets of cells that have at least one associated object. This step is executed in parallel with respect to the grids. The notebook writes its output to the `data_simulator` folder. Readers can skip executing this notebook, as we already provide the candidates in a set of pickle files (to this end, see the dataset repository).

- **6 - Filter and flatten candidates.ipynb**: this notebook prepares the candidates generated by the previous notebook for spatial scan statistics processing. More precisely, it flattens the various data structures that contain the candidates. The notebook writes its output to the `data_simulator` folder. Readers can skip executing this notebook, as we already provide the data structures containing the flattened candidates in a pickle file (to this end, see the dataset repository).

This concludes the notebooks that implement the first four steps of our approach, i.e., the steps that focus on preparing the simulator's movement data for the subsequent steps.

***

### Run the experimental evaluation

Next, we perform the **experimental evaluation** by executing the following notebooks.

- **Exp - Generator of unfair auditable datasets.ipynb**: this notebook generates the various groups of unfair auditable datasets required for the experimental evaluation. To configure the generation process, the user must customize the `params_datasets` variable in the notebook, which is a dictionary containing the various parameters that configure the injection procedure. Although the generation process is massively parallelized with respect to the number of datasets to be generated in a group, some groups of datasets still require a very long time to be generated. Therefore, if a user wishes to execute this notebook to generate their own unfair auditable datasets, we recommend using a machine with a large number of CPU logical cores. The notebook writes its output to the `experiments` folder. In any case, readers can skip executing this notebook, as we already provide the unfair datasets used during the experimental evaluation (to this end, see the dataset repository).

- **7 - Assess Datasets for Fairness Based on Movement Patterns.ipynb**: this notebook performs the Bernoulli-based spatial scan statistic considering all the candidate subsets of cells from the various materialized grids over groups of unfair auditable datasets. The execution can be customized by modifying the following variables: `num_simulations`, which represents the number of Monte Carlo simulations used to approximate the distribution of the maximum log-likelihood ratio under the null hypothesis; `alpha`, which represents the required statistical significance level; and `list_set_groups`, which specifies the set of dataset groups to consider (e.g., the groups in which we vary the number of objects associated with a hotspot). The notebook outputs the results to the `experiments` folder, more precisely to the subfolders containing the sets of dataset groups being considered. To facilitate reuse, we have already included the files containing the results of this step, which were subsequently used in the article (to this end, see the dataset repository).

At this point, we have the results of the Bernoulli-based spatial scan statistic applied to several groups of unfair auditable datasets. Next, we execute the set of notebooks that transform these results into the ones actually shown in the article.

- **8 - Compute stats results group datasets.ipynb**: this notebook transforms the results of the Bernoulli-based spatial scan statistic into formats (i.e., LaTeX tables and CSV files) suitable for the article. Note that the user must customize the `name_set_groups_datasets` variable, which specifies the particular set of dataset groups considered by the notebook (e.g., the set in which we vary the number of objects per hotspot). To facilitate reuse, we have included the CSV files that were subsequently used to generate the article's plots (to this end, see the dataset repository).

- **9 - Plot results.ipynb**: this notebook generates the plots from the CSVs generated by the previous notebook.

***


### Exploratory strategy example

The notebook **10 - Advanced inspection hypothesis test results.ipynb** makes it possible to perform the exploratory strategy described in the article's experimental evaluation on any auditable datasets. More precisely: assuming that there is a group of unfair auditable datasets and the results of the assessments conducted on it by the notebook **7**, both located in an appropriate path, the user can explore on a map the assessment results for any of the dataset in the group and compare what the approach has found w.r.t. the true hotspots (if any). The user can customize the notebook by changing the following variables: `name_set_groups_datasets`, which specifies the set of dataset groups considered; `idx_selected_group`, which specifies the dataset group selected from that set; and `idx_dataset`, which specifies the index of the dataset selected from the group.
