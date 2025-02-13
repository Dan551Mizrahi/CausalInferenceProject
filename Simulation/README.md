# Simulation of Traffic Scenarios for Data Generation
This folder contains code for simulating traffic scenarios and generating data for causal inference analysis. We use the SUMO traffic simulation platform to simulate traffic scenarios and generate data for evaluating the impact of traffic light control policies on traffic flow.

## Directory Contents
- **/**
  - `run_simulation.py` - A script for running traffic simulations and generating data.
  - **results_utils/** - A directory containing utility scripts for parsing simulation results.
    - `exp_results_parser.py` - Description: A script for parsing a single simulation results. It contains a class that parses the output of a SUMO simulation and extracts relevant data for causal inference analysis, and also functions for simple output parsing.
  - **run_logs/** - A directory for storing logs generated during the simulation runs, mainly for saving the configuration of each run.
  - **SUMO/** - A directory containing code for interfacing with the SUMO traffic simulation platform.
    - `SUMOAdapter.py` - Description: A script for interfacing with the SUMO simulation environment using a SUMOAdapter class.
    - `TL_policy.py` - Description: An implementation of the traffic engineer treatment assignment policy.
    - **cfg_files/** - A directory containing configuration files generated for the SUMO simulation.
    - **outputs/** - A directory for storing simulation outputs, including tripinfo and tripinfo.xml files.
    - **template_files/** - A directory containing template files for generating SUMO configuration files, including network, route, and additional files.

## How to Extend
To extend the simulation capabilities, you can modify the existing simulation scripts or create new ones. You can also add new traffic scenarios, traffic light control policies, or other simulation parameters to generate different types of data for causal inference analysis. Some examples of simple extensions include:
- Modifying the simulation parameters by giving arguments using the command line or by modifying the `argparse_utils.py` script. The parameters include the mean demand size, the length of the simulation, and number of runs/experiments. Also, a gui can easily be used to view the simulation using the gui=True flag.
- Creating new policies for choosing a treatment assignment can be done by modifying the `TL_policy.py` script. A new policy can be implemented as a function that takes the simulation state as input and returns the treatment assignment (T=0 or T=1 or T=2).
- Other SUMO built-in traffic light control policies can be implemented by creating more templates in the `template_files` directory and modifying the `TL_policy.py` script to use the new templates.
- Other network files can be created by modifying the `template_files` directory and the `SUMOAdapter.py` script to use the new network files. Note that the names of the junctions should also be modified in the SUMOAdapter.py script.
- More output analysis can be done by modifying the `exp_results_parser.py` script to extract more data from the simulation outputs. This can include extracting more vehicle data, traffic flow data, or other relevant information for causal inference analysis.
- For more extensive modifications, you can create new scripts or directories to handle different aspects of the simulation, such as vehicle behavior, traffic flow dynamics, or other simulation components. Fell free to reach out to the authors for more information :)