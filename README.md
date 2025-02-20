# A Modular Traffic Dataset Generation Framework for Causal Inference Evaluation

This repository contains code for simulating traffic scenarios and applying causal inference algorithms to evaluate the impact of traffic light control policies on traffic flow.

## Repository Overview

This project investigates the causal effect of different traffic light control policies on average vehicle duration in a simulated traffic environment. We use the SUMO traffic simulation platform to generate data and apply various causal inference methods to estimate the Average Treatment Effect (ATE) of different policies.

## Repository Contents

- **/**
  - `main.py` - Description: A run-all script that executes the entire pipeline from data generation to causal inference methods (fit and predict) and results analysis.
2  - **Figures/** - A directory containing graphs about the data and the performance evaluation of CI models.
  - `models.py` - Description: classes and methods to create and evaluate CI methods using <code>causallib</code>
  - 'stats_and_preprocessing.ipynb' - Jupyter notebook for basic EDA and some tests.
- **Figures/** - A directory for storing figures generated during the analysis, included in the report.
- **Simulation/**
  - `run_simulation.py` - A script for running traffic simulations and generating data.
  - **results_utils/** - A directory containing utility scripts for parsing simulation results.
    - `exp_results_parser.py` - Description: A script for parsing a single simulation results. 
    - `results_parse.py` - Description: A helper script for parsing multiple simulation results.
  - **run_logs/** - A directory for storing logs generated during the simulation runs, mainly for saving the configuration of each run.
  - **run_utils/** - A directory containing utility scripts for running simulations.
    - `argparse_utils.py` - Description: A script for parsing command-line arguments for the simulation.
    - `TL_policy.py` - Description: An implementation of the traffic engineer treatment assignment policy.
  - **SUMO/** - A directory containing code for interfacing with the SUMO traffic simulation platform.
    - `SUMOAdapter.py` - Description: A script for interfacing with the SUMO simulation environment using a SUMOAdapter class.
    - **cfg_files/** - A directory containing configuration files generated for the SUMO simulation.
    - **outputs/** - A directory for storing simulation outputs, including tripinfo and tripinfo.xml files.
    - **template_files/** - A directory containing template files for generating SUMO configuration files, including network, route, and additional files.
  - **Simulated_Data/** - A directory containing simulated traffic data, both for ground truth and counterfactual scenarios.


## Dependencies

*   SUMO 1.19.0
*   Python libraries as appear in the requirements.txt file.

## How to Run

1.  Install dependencies.
2.  Configure simulation parameters in `simulation/config.py`.
3.  Run simulations using `simulation/run_simulations.py`.
4.  Analyze data and run causal inference algorithms using scripts in `causal_inference/`.

## Results

Results of the causal inference analysis are presented in the `results/` directory.

## Authors

*   Tal Kraicer
*   Dan Shlomo Mizrahi

## Acknowledgments

This project is based on the work described in the following paper:

Dorie, V., Hill, J., Shalit, U., Scott, M., & Cervone, D. (2019). Automated versus do-it-yourself methods for causal inference: Lessons learned from a data analysis competition. Statistical Science, 43-68.

Also, the initial idea for this project was founded under Intro to Causal Inference course at the Technion, taught by Prof. Uri Shalit and Mr. Rom Gutman