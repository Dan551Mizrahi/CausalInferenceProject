import Simulation.run_simulation as run_simulation
import CI_competition.run_competition as run_competition
from ATE_calculator.bootstrap_ATE import *
from argparse_utils import get_args
from Figures.competition_figures.ate_tables import export_ate_table_csv
import os
import pandas as pd

simulation_data_dir = "Simulated_Data"
training_data_filename = "Training_data"
testing_data_filename = "Testing_data"


def main():
    run_args, simulation_arguments, competition_args = get_args()

    if run_args["run_simulation"]:
        # Running the simulation
        print("Running simulation")
        training_df, testing_df = run_simulation.main(simulation_arguments, run_args)

        # Saving simulation data
        os.makedirs(simulation_data_dir, exist_ok=True)

        for i in range(run_args["num_runs"]):
            run_data_dir = os.path.join(simulation_data_dir, f"run_{i}")
            os.makedirs(run_data_dir, exist_ok=True)

            training_df_i = training_df.iloc\
                [i * run_args["num_experiments"]:
                 (i + 1) * run_args["num_experiments"]]
            training_df_i.reset_index(drop=True, inplace=True)
            training_df_i.to_pickle(f"{run_data_dir}/{training_data_filename}.pkl")

            testing_df_i = testing_df.iloc\
                [i * run_args["num_experiments"]:
                    (i + 1) * run_args["num_experiments"]]
            testing_df_i.reset_index(drop=True, inplace=True)
            testing_df_i.to_pickle(f"{run_data_dir}/{testing_data_filename}.pkl")

            # Calculating testing ATEs
            testing_ATEs = calculate_ATEs(testing_df_i)
            testing_ATEs.to_pickle(f"{run_data_dir}/testing_ATEs.pkl")
            # T0DO: Verify the bootstrap
            testing_ATEs_bootstrap = bootstrap_ATEs(testing_df_i)
            testing_ATEs_bootstrap.to_pickle(f"{run_data_dir}/testing_ATEs_bootstrap.pkl")

            # calculating training ATEs
            training_ATEs = calculate_ATEs(training_df_i)
            training_ATEs.to_pickle(f"{run_data_dir}/training_ATEs.pkl")

    if run_args["run_competition"]:
        for i in range(run_args["num_runs"]):
            # Running the competition
            print("Running competition")
            run_data_dir = os.path.join(simulation_data_dir, f"run_{i}")
            training_df = pd.read_pickle(f"{run_data_dir}/{training_data_filename}.pkl")
            run_competition.main(competition_args, run_args, training_df, i)

    if run_args["parse_results"]:
        export_ate_table_csv(run_args["num_runs"], "Figures/ATEs.xlsx")







if __name__ == '__main__':
    main()
