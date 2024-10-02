# Causal Inference in Traffic Simulations

This repository contains code for simulating traffic scenarios and applying causal inference algorithms to evaluate the impact of traffic light control policies on traffic flow.

## Project Overview

This project investigates the causal effect of different traffic light control policies on average vehicle duration in a simulated traffic environment. We use the SUMO traffic simulation platform to generate data and apply various causal inference methods to estimate the Average Treatment Effect (ATE) of different policies.

## Repository Contents

*   **simulation/:** Contains code for running traffic simulations in SUMO.
*   **causal_inference/:** Contains implementations of various causal inference algorithms.
*   **data/:** Contains sample simulation data.
*   **results/:** Contains results of causal inference analysis.

## Dependencies

*   SUMO
*   Python libraries: numpy, pandas, scikit-learn, causalgraphicalmodels, ...

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
