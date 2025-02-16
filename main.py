import Simulation.run_simulation as run_simulation
import CI_competition.run_competition as run_competition
from argparse_utils import get_args
from Figures.competition_figures.ate_tables import export_ate_table_excel
import os
import pandas as pd

simulation_data_dir = "Simulation/simulated_data"
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
            run_data_dir = os.path.join(simulation_data_dir, f"run_{i+run_args['continue_from']}")
            training_df = pd.read_pickle(f"{run_data_dir}/{training_data_filename}.pkl")
            run_competition.main(competition_args, run_args, training_df, i+run_args['continue_from'])

    if run_args["parse_results"]:
        export_ate_table_excel(run_args["num_runs"], "Figures/basic_ATEs_table.xlsx")
        export_ate_table_excel(run_args["num_runs"], "Figures/relative_error_ate_table.xlsx", type="paired", error_function="relative_error")
        export_ate_table_excel(run_args["num_runs"], "Figures/rmse_ate_table.xlsx", type="paired", error_function="squared_error",
                               agg_function="rooted_mean")


if __name__ == '__main__':
    main()

