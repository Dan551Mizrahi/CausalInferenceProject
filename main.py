import Simulation.run_simulation as run_simulation
import CI_competition.run_competition as run_competition
from argparse_utils import get_args
from Figures.competition_figures.ate_tables import export_ate_table_csv
import os
import pandas as pd

simulation_data_dir = "simulated_data"
training_data_filename = "training_data"
testing_data_filename = "testing_data"


def main():
    run_args, simulation_arguments, competition_args = get_args()

    if run_args["run_simulation"]:
        # Running the simulation
        print("Running simulation")
        training_df, testing_df = run_simulation.main(simulation_arguments, run_args)

    if run_args["run_competition"]:
        for i in range(run_args["num_runs"]):
            # Running the competition
            print("Running competition")
            run_data_dir = os.path.join("Simulation", simulation_data_dir, f"run_{i}")
            training_df = pd.read_pickle(f"{run_data_dir}/{training_data_filename}.pkl")
            run_competition.main(competition_args, run_args, training_df, i)

    if run_args["parse_results"]:
        export_ate_table_csv(run_args["num_runs"], "Figures/ATEs.xlsx")


if __name__ == '__main__':
    main()
