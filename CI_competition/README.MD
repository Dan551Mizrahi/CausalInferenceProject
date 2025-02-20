# Running a Causal Inference Pipeline on Traffic Simulation Data

This folder contains code for applying causal inference algorithms to estimate the impact of traffic light control policies on traffic flow (ATE).

## Directory Contents
- **/**
-`run_competition.py` - A script for running causal inference algorithms on traffic simulation data.
-`utils.py` - A script containing utility functions for data processing and evaluation.
- **data/** - A directory containing data generated from traffic simulations.
- **ATEs/** - A directory for storing the estimated ATEs from different causal inference algorithms, including the true ATE, over multiple runs.
- **estimators/** - A directory containing different causal inference algorithms for estimating the ATE.
  - `CausalInferenceEstimationModel.py` - A script containing the base class for causal inference estimation models. All causal inference models should inherit from this class and implement the `estimate_ATE` method.
  - `models_definitions.py` - A script containing the definitions of different causal inference models, meaning the different sets of hyperparameters for each model.
  - `IPWModel.py` - A script containing the implementation of the Inverse Probability Weighting (IPW) model for estimating the ATE.
  - `MatchingModel.py` - A script containing the implementation of the Matching model for estimating the ATE.
  - `PropensityMatchingModel.py` - A script containing the implementation of the Propensity Matching model for estimating the ATE.
  - `StandradizationModel.py` - A script containing the implementation of the Standardization model for estimating the ATE.
  - `TMLEModel.py` - A script containing the implementation of the Targeted Maximum Likelihood Estimation (TMLE) model for estimating the ATE.

## Output
After running the `run_competition.py` script, the estimated ATEs from different causal inference algorithms are stored in the `ATEs` directory. The output includes a directory for each run, with the following files:
- A `<model>.pkl` file containing the estimated ATE from the causal inference model `<model>`.
- A `True.pkl` file containing the true ATE from the simulation data.

## How to Extend
To extend the causal inference capabilities, you can modify the existing causal inference models or create new ones.
- You can modify the existing causal inference models by changing the hyperparameters using the `models_definitions.py` script.
- You can create new causal inference models by implementing the `estimate_ATE` method and inheriting from the `CausalInferenceEstimationModel` class. Then, just place the new model in the `estimators` directory.
- For other changes such as calculating ATT instead of ATE, you can modify the `CausalInferenceEstimationModel.py` script to include the new method and update the `run_competition.py` script to use the new method.

