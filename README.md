# A Modular Traffic Dataset Generation Framework for Causal Inference Evaluation

This repository contains code for simulating traffic scenarios and applying causal inference algorithms to evaluate the
impact of traffic light control policies on traffic flow.

## Repository Overview

This project investigates the causal effect of different traffic light control policies on average vehicle delay in a
simulated traffic environment. We use the SUMO traffic simulation platform to generate data and apply various causal
inference methods to estimate the Average Treatment Effect (ATE) of different policies.

## Repository Contents

- **/**
    - `main.py` - Description: A run-all script that executes the entire pipeline from data generation to causal
      inference methods (fit and predict) and results analysis.
    - `argparse_utils.py` - Description: Arguments reader to all parts of the pipeline. Contains the arguemnts for the
      simulation (generated data), causal inference methods, and results analysis. It also contains general arguments
      for the run such as the number of processors.
- **Simulation/** - A directory containing all necessary code for data generation.
- **CI_competition/** - A directory containing all necessary code for causal inference methods.
- **Figures/** - A directory containing all figures generated to compare the causal inferences method's performance.

Each directory contains its own README file inside of it, including extension possibilities and more detailed
information.

## Dependencies

* SUMO 1.19.0
* Python libraries as appear in the requirements.txt file.

## How to Run

1. Install dependencies.
2. Configure parameters in `argparse_utils.py` (or use the default values).
3. Run the main script: `python main.py`. <br>
   **Note** - The simulation part of the pipeline is time-consuming and requires a powerful machine. It is recommended
   to run the simulation part on a machine with multiple processors or reduce the number of runs or number of
   processors.

## Results

The results for each component of the pipeline will appear inside its respective directory, as described in the README
file of each directory.

## Authors

* Tal Kraicer
* Dan Shlomo Mizrahi