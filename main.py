import Simulation.run_simulation as run_simulation
import CI_competition.run_competition as run_competition
from argparse_utils import get_args
from Figures.run_figures import main_figures

simulation_data_dir = "Simulation/simulated_data"
training_data_filename = "training_data"
testing_data_filename = "testing_data"


def main():
    run_args, simulation_arguments, competition_args = get_args()
    run_args["simulation_data_dir"] = simulation_data_dir
    run_args["training_data_filename"] = training_data_filename
    run_args["testing_data_filename"] = testing_data_filename

    if run_args["run_simulation"]:
        # Running the simulation
        print("Running simulation")
        training_df, testing_df = run_simulation.main(simulation_arguments, run_args)

    if run_args["run_competition"]:
        # Running the competition
        print("Running competition")
        run_competition.main(competition_args, run_args)

    if run_args["parse_results"]:
        main_figures(run_args["num_runs"])

if __name__ == '__main__':
    main()

