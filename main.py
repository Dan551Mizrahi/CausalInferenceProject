import Simulation.run_simulation as run_simulation
import CI_competition.run_competition as run_competition
from ATE_calculator.bootstrap_ATE import *
from argparse_utils import get_args
import os
import pandas as pd

simulation_data_dir = "Simulated_Data"
training_data_filename = "Training_data.pkl"
testing_data_filename = "Testing_data.pkl"


def main():
    run_args, simulation_arguments, competition_args = get_args()

    if run_args["run_simulation"]:
        # Running the simulation
        print("Running simulation")
        training_df, testing_df = run_simulation.main(simulation_arguments, run_args)

        # Saving simulation data
        os.makedirs(simulation_data_dir, exist_ok=True)
        training_df.to_pickle(f"{simulation_data_dir}/{training_data_filename}")
        testing_df.to_pickle(f"{simulation_data_dir}/{testing_data_filename}")

        # Calculating testing ATEs
        testing_ATEs = calculate_ATEs(testing_df)
        testing_ATEs.to_pickle(f"{simulation_data_dir}/testing_ATEs.pkl")
        # T0DO: Verify the bootstrap
        testing_ATEs_bootstrap = bootstrap_ATEs(testing_df)
        testing_ATEs_bootstrap.to_pickle(f"{simulation_data_dir}/testing_ATEs_bootstrap.pkl")

        # calculating training ATEs
        training_ATEs = calculate_ATEs(training_df)
        training_ATEs.to_pickle(f"{simulation_data_dir}/training_ATEs.pkl")

    if run_args["run_competition"]:
        # Running the competition
        print("Running competition")
        training_df = pd.read_pickle(f"{simulation_data_dir}/{training_data_filename}")
        run_competition.main(competition_args, run_args, training_df)





if __name__ == '__main__':
    main()
