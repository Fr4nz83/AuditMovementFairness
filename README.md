# Assessing Predictive Models for Fairness Based on Movement Patterns

This repository contains the code behind the paper "Assessing Predictive Models for Fairness Based on Movement Patterns".

In this repository you will find:

- Indications on how to generate synthetic movement data with the Patterns of Life simulator;
- The source code used to implement the assessment approach described in the paper;
- The source code used to generate the synthetic unfair auditable datasets used in the experimental evaluation;
- The source code used to conduct the experimental evaluation.


## How to generate the synthetic movement data

We generate a dataset of syntetic trajectories with the Patterns of Life simulator from https://github.com/onspatial/generate-mobility-dataset. The simulator's configuration has been modified such that it generates the movement data of 100,000 agents moving within the city of Atlanta, Georgia, USA, with a sampling rate of 2 minutes. The seed that has been used for reproducibility purposes has been set to 2.

To facilitate the reader, we made available the dataset used in the experimental evaluation in the parquet file format, which can be opened with the Python pandas library. Although compressed, the file is very large, hence we recommend to open it in a system with enough RAM.


## How to set up the Python environment to run our assessment approach

In our repository we made available a YAML file that specifies all the dependencies needed to set up an appropriate conda environment. Assuming that you have installed conda, please run ```conda env create -f geo.yaml''', which will create the "geo" environment.


## How to run our assessment approach


