import os
from tqdm import tqdm
from multiprocessing import Pool
from Simulation.SUMO.SUMOAdapter import SUMOAdapter
from Simulation.SUMO.TL_policy import determine_policy
from Simulation.results_utils.exp_results_parser import *
from main import simulation_data_dir, training_data_filename, testing_data_filename
from ATE_calculator.bootstrap_ATE import *


def save_results(training_df, testing_df, num_runs):
    # Saving simulation data
    os.makedirs(simulation_data_dir, exist_ok=True)

    # splitting the data into num_runs
    for i in range(num_runs):
        run_data_dir = os.path.join(simulation_data_dir, f"run_{i}")
        os.makedirs(run_data_dir, exist_ok=True)

        training_df_i = training_df.iloc[i * num_runs:(i + 1) * num_runs]
        training_df_i.reset_index(drop=True, inplace=True)
        training_df_i.to_pickle(f"{run_data_dir}/{training_data_filename}.pkl")

        testing_df_i = testing_df.iloc[i * num_runs:(i + 1) * num_runs]
        testing_df_i.reset_index(drop=True, inplace=True)
        testing_df_i.to_pickle(f"{run_data_dir}/{testing_data_filename}.pkl")

        # Calculating testing ATEs
        testing_ATEs = calculate_ATEs(testing_df_i)
        testing_ATEs.to_pickle(f"{run_data_dir}/testing_ATEs.pkl")

        # TODO: Verify the bootstrap
        testing_ATEs_bootstrap = bootstrap_ATEs(testing_df_i)
        testing_ATEs_bootstrap.to_pickle(f"{run_data_dir}/testing_ATEs_bootstrap.pkl")

        # calculating training ATEs
        training_ATEs = calculate_ATEs(training_df_i)
        training_ATEs.to_pickle(f"{run_data_dir}/training_ATEs.pkl")


def simulate(simulation_arguments):
    sumo = SUMOAdapter(**simulation_arguments)

    # run with T=0
    sumo.init_simulation()
    sumo.run_simulation()

    # determine policy
    delay_sum = get_delay_sum(sumo.tripinfo_file)
    policy = determine_policy(delay_sum)

    # run with determined policy
    new_seed = simulation_arguments["seed"] + 1
    sumo.re_init_simulation(seed=new_seed, TL_type=policy, chosen=True)
    sumo.run_simulation()
    training_row = create_row(ResultsParser(sumo.tripinfo_file), delay_sum, policy)

    testing_rows = []

    # For true ATE calculation
    sumo.re_init_simulation(TL_type=0)
    sumo.run_simulation()
    testing_rows.append(create_row(ResultsParser(sumo.tripinfo_file), delay_sum, 0))
    sumo.re_init_simulation(TL_type=1)
    sumo.run_simulation()
    testing_rows.append(create_row(ResultsParser(sumo.tripinfo_file), delay_sum, 1))
    sumo.re_init_simulation(TL_type=2)
    sumo.run_simulation()
    testing_rows.append(create_row(ResultsParser(sumo.tripinfo_file), delay_sum, 2))

    return training_row, testing_rows


def main(simulation_arguments, run_args):
    # Running the simulation simultaneously
    with Pool(run_args["num_processes"]) as p:
        results = list(tqdm(p.imap(simulate, simulation_arguments), total=len(simulation_arguments)))

    # create training and testing dataframes
    training_table, testing_table = [], []
    for training_row, testing_rows in results:
        training_table.append(training_row)
        testing_table += testing_rows

    training_df = pd.DataFrame(training_table)
    testing_df = pd.DataFrame(testing_table)

    # Saving simulation results to different files
    save_results(training_df, testing_df, run_args["num_runs"])

    return training_df, testing_df

if __name__ == '__main__':
    arguments = {"seed": 1304804, "demand":163, "episode_len":600, "lane_log_period":60, "gui":True}
    simulate(arguments)